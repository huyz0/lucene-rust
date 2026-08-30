# M2 sweep — batch `b2-packed`

Rust files swept (every `fn`), against Lucene 10.5.0 at `/home/tuong/work/lucene`:

| Rust | Java counterpart(s) |
|---|---|
| `crates/lucene-codecs/src/for_util.rs` | `codecs/lucene104/ForUtil.java`, `codecs/lucene104/PForUtil.java`, `internal/vectorization/PostingDecodingUtil.java`, `internal/vectorization/DefaultVectorUtilSupport.expand8`, `util/packed/PackedInts.bitsRequired/unsignedBitsRequired` |
| `crates/lucene-codecs/src/packed_ints.rs` | `util/packed/PackedInts.Format.PACKED`, `util/packed/BulkOperationPacked.java` (byte `encode`/`decode`) |
| `crates/lucene-codecs/src/block_packed.rs` | `util/packed/AbstractBlockPackedWriter.java`, `util/packed/BlockPackedWriter.java`, `util/packed/BlockPackedReaderIterator.java` |
| `crates/lucene-codecs/src/direct_reader.rs` | `util/packed/DirectReader.java`, `util/packed/DirectWriter.java` |
| `crates/lucene-codecs/src/direct_monotonic.rs` | `util/packed/DirectMonotonicReader.java`, `util/packed/DirectMonotonicWriter.java` |
| `crates/lucene-codecs/src/indexed_disi.rs` | `codecs/lucene90/IndexedDISI.java` |

Note on the postings codec: `lucene103/` in this checkout contains only `blocktree`;
`ForUtil`/`PForUtil`/`PostingIndexInput`/`PostingsUtil` live in `lucene104/`, which is
what `docs/parity.md` targets and what this sweep compared against.
`PostingIndexInput` is a benchmarking wrapper around `ForUtil.decode` with no
state of its own — this port's `ForUtil::decode` takes the `DataInput` directly,
so there is nothing to port (`INTENTIONAL`, recorded, no finding).

---

## `crates/lucene-codecs/src/for_util.rs`

Java: `codecs/lucene104/ForUtil.java`, `codecs/lucene104/PForUtil.java`.

| Rust `fn` | Java | Verdict |
|---|---|---|
| `mask32` / `mask16` / `mask8` / `expand_mask8` / `expand_mask16` | `ForUtil.mask32/16/8`, `expandMask8/16` | identical (Rust additionally defines `mask32(32) == u32::MAX`, where Java's `MASKS32` table stops at index 31 — see finding 4) |
| `expand8` / `expand16` / `collapse8` / `collapse16` | `VectorUtil.expand8` (`DefaultVectorUtilSupport.expand8`), `ForUtil.expand16/collapse8/collapse16` | identical |
| `split_ints` | `PostingDecodingUtil.splitInts` | identical, incl. the signed `(bShift - 1) / dec` intermediate |
| `decode1`..`decode16`, `decode_slow` | `ForUtil.decode1..decode16`, `decodeSlow` | identical (spot-checked every `splitInts` argument tuple and every tail loop) |
| `ForUtil::decode` | `ForUtil.decode` | divergent — finding 2 |
| `ForUtil::pfor_decode` / `pfor_decode` | `PForUtil.decode` | identical |
| `num_bytes` | `ForUtil.numBytes` | identical |
| `encode_generic` | `ForUtil.encode(int[],int,int,DataOutput,int[])` | identical algorithm; scratch ownership divergent — finding 3 |
| `for_encode` / `ForUtil::encode` | `ForUtil.encode(int[],int,DataOutput)` | was divergent (copied its input) — finding 3 |
| `bits_required` | `PackedInts.bitsRequired(int)` / `unsignedBitsRequired(int)` | was divergent — finding 1 |
| `all_equal` | `PForUtil.allEqual` | identical |
| `pfor_encode` / `pfor_encode_with` | `PForUtil.encode` | identical after finding 1 |
| — | `PForUtil.skip` | **not ported** — finding 5 |

### 1. `[CORRECTNESS]` `bits_required(0)` returned 0; Java's returns 1

- **Java**: `PackedInts.unsignedBitsRequired` is `Math.max(1, 32 - numberOfLeadingZeros(bits))`
  and its javadoc says "NOTE: This method returns at least 1". `PForUtil.encode` feeds
  that into `histogram[bits]++`.
- **We did**: `32 - v.leading_zeros()`, i.e. `0` for `v == 0` — and the module doc
  asserted this matched Java, which it does not.
- **Consequence**: `PForUtil`'s width search walks `b` down from `maxBitsRequired`
  while `count(bits > b) <= 7`. With zero mapped to bucket 0 instead of bucket 1,
  `b` could reach 0 for any block whose *non-zero* values number at most 7 and fit
  in 8 bits; Java stops at `b == 1` because every value contributes to
  `count(bits > 0) == 256`. Worked example: 254 zeros, one `1`, one `200` — Java
  writes a 35-byte block (token `0x21`, a 1-bit packed body, one exception);
  we wrote a 6-byte all-equal/vint block. Both decode to the same 256 values and
  real Lucene reads either, so this was a byte-fidelity divergence rather than a
  data-loss one — but it is exactly the class of difference a byte-level
  differential test against a Java-written `.doc` would trip over, and the write
  path is meant to be byte-compatible.
- **Fixed**: `bits_required` now returns `(32 - lz).max(1)`.
  Tests: `bits_required_never_returns_zero_like_javas_packed_ints`,
  `pfor_encode_mostly_zero_block_picks_javas_bit_width` (pins the exact token,
  length and exception bytes of the worked example above).

### 2. `[CORRECTNESS]` `ForUtil::decode` panicked on an out-of-range `bits_per_value`

- **Java**: no explicit check; `decodeSlow` indexes `MASKS32` (`new int[32]`), so
  anything outside `1..=31` throws `ArrayIndexOutOfBoundsException`.
- **We did**: `decode_slow` computed `num_ints = bits * 8` and called
  `ints.split_at_mut(num_ints.max(1))` — for `bits >= 33` that is a slice-index
  **panic**, and for `bits == 0` the tail loop underflowed `bits_per_value - remaining_bits`
  (another panic). `postings::read_full_block_header` reads this width as a signed
  byte straight off disk (`if bits_per_value_byte > 0`), so a corrupt `.doc` reaches
  it with up to 127.
- **Consequence**: a panic where every other path in this decoder reports corruption
  through `Result` (and where `lucene-ffi`'s `catch_unwind` is the only thing between
  it and the JVM).
- **Fixed**: `ForUtil::decode` rejects anything outside `1..=32` with
  `Error::Corrupted`. Test: `decode_rejects_bits_per_value_outside_the_supported_range`.

### 3. `[PERF]` `for_encode` copied its input; `encode_generic` zero-filled a fresh 1 KiB scratch per block

- **Java**: `ForUtil.encode(int[] ints, int bitsPerValue, DataOutput out)` collapses
  lanes **in place** in the caller's array and packs through `private final int[] tmp`,
  allocated once per `ForUtil` instance and never cleared (the first loop assigns
  rather than ORs, so stale content is harmless).
- **We did**: `for_encode(&[u32; 256], ...)` copied the caller's block into a local
  array, and `encode_generic` declared `let mut tmp = [0u32; BLOCK_SIZE]` — a 1 KiB
  memcpy plus a 1 KiB memset per encoded block that Java pays neither of. (This is
  the encode-side twin of the decode-side scratch finding already fixed in `ForUtil`'s
  doc comment history.)
- **Fixed**: `for_encode`/`ForUtil::encode` take `&mut [u32; BLOCK_SIZE]` and collapse
  in place, matching Java's contract exactly; `encode_generic` takes the scratch by
  reference and `ForUtil::encode` passes the instance's own buffer. Call sites updated:
  `postings_writer::write_full_block`, `for_util`/`postings` tests, the bench.
- **Measurement, honestly**: the copy removal **cannot** be isolated in a
  microbenchmark, because a repeatable bench of a destructive encoder must restore
  its input each iteration — which reinstates exactly the copy that was removed. The
  benefit is structural: `postings_writer` builds `deltas`/`freqs` fresh per block and
  no longer needs a second 1 KiB duplicate of it. A new `for_util/for_encode` group in
  `benches/for_util_decode.rs` measures `oneshot` (fresh `ForUtil` per block, what
  `postings_writer` does today) against `reused` (instance held across blocks, what
  Lucene does): absolute cost is 64–470 ns per 256-value block across
  `bits ∈ {1,5,8,12,16,24,31}`, and the two arms did **not** separate above the noise
  floor on this machine (five other sweep batches were compiling concurrently;
  run-to-run spread on a single arm reached ±20%). Recorded as an open item rather
  than claimed as a win — see *Open items*.

### 4. `[INTENTIONAL]` `bits_per_value == 32` is decodable here and not in Java

`mask32(32)` returns `u32::MAX` here; Java's `MASKS32` has 32 entries, so
`decodeSlow(32, ...)` throws. `for_encode`/`decode` handle 32 correctly and
round-trip, and the existing bench covers it. This is a strict superset with no
reachable writer (the `PForUtil` token holds `bitsPerValue` in 5 bits, so 32 would
alias to the all-equal marker `0`). Kept, and `pfor_encode` now carries a
`debug_assert!(max_bits_required <= 31)` so the documented 31-bit precondition is
checked where it lives rather than only described.

### 5. `[MISSING]` `PForUtil.skip` is not ported

Java's `skip(DataInput)` reads the token and jumps
`numBytes(bitsPerValue) + (numExceptions << 1)` bytes without decoding.
**Not fixed**: no caller can reach it — this port never skips a `PForUtil` block
without decoding it (`postings::LazyDocsCursor` skips at the *level-0 block* level
using the on-wire block length, which is a different mechanism). `num_bytes`, the
one building block it would need, is already present and tested. Adding an
uncallable public function is worse than recording the gap.

### Verdict

Swept clean, with findings 1–3 fixed and 4–5 recorded.

---

## `crates/lucene-codecs/src/packed_ints.rs`

Java: `util/packed/PackedInts.Format.PACKED.byteCount`, `util/packed/BulkOperationPacked.java`
(the `byte[]` `encode`/`decode` overloads — the MSB-first contiguous bitstream).

| Rust `fn` | Java | Verdict |
|---|---|---|
| `get` | `BulkOperationPacked.decode(byte[], ...)`, one value at a time | identical bit layout |
| `byte_count` | `PackedInts.Format.PACKED.byteCount` | identical (`ceil(count*bits/8)`; Java's `Math.ceil` over a `double` is exact for the whole legal domain, `valueCount <= 2^31` and `bitsPerValue <= 64` give a product under 2^37) |
| `encode` | `BulkOperationPacked.encode(long[], byte[], ...)` | identical bit layout |

No findings. `bits_per_value == 0` is accepted here (writes nothing, reads 0) where
Java's `BulkOperation.of` would index `packedBulkOps[-1]`; that width is handled by
`block_packed`'s own `bitsPerValue == 0` branch before it ever reaches here, exactly
as in Java, and the behaviour is the natural extension of the same formula.

### Verdict

Swept clean.

---

## `crates/lucene-codecs/src/block_packed.rs`

Java: `util/packed/BlockPackedWriter.flush`, `util/packed/AbstractBlockPackedWriter`
(`writeVLong`, `writeValues`), `util/packed/BlockPackedReaderIterator.refill`/`readVLong`.
Block size is fixed at 64 here; `Lucene90CompressingTermVectorsWriter.PACKED_BLOCK_SIZE`
is 64 and term vectors is the only user, so the parameterisation Java carries has no
second value to take (`INTENTIONAL`).

| Rust `fn` | Java | Verdict |
|---|---|---|
| `decode_all` | `BlockPackedReaderIterator.refill`, driven to completion | identical after finding 7 |
| `read_min_value_vlong` (new) | `BlockPackedReaderIterator.readVLong` | added — finding 7 |
| `write_min_value_vlong` (new) | `AbstractBlockPackedWriter.writeVLong` | added — finding 6 |
| `unsigned_bits_required` (new) | `PackedInts.unsignedBitsRequired` | added — finding 6 |
| `max_value` (new) | `PackedInts.maxValue` | added — finding 6 |
| `encode_all` | `BlockPackedWriter.flush` + `AbstractBlockPackedWriter.writeValues` | was divergent — finding 6 |
| — | `skip`, `next()`, `next(count)`, `ord()`, `reset()` | not ported, `INTENTIONAL` (decode-once design, documented in the module header) |

### 6. `[CORRECTNESS]` `encode_all` mis-encoded any block containing a negative value

- **Java** (`BlockPackedWriter.flush`): computes the block's `min` and `max`,
  `delta = max - min`, `bitsRequired = delta == 0 ? 0 : unsignedBitsRequired(delta)`;
  then `min = 0` if `bitsRequired == 64` (raw, no delta-encoding) else, if `min > 0`,
  `min = max(0, max - maxValue(bitsRequired))` to shorten the `minValue` varint;
  writes the token, the zigzag varint `minValue` when non-zero, then the
  `value - min` deltas.
- **We did**: hardcoded `minValue = 0` and derived `bitsPerValue` from the block
  **max** alone (`if max <= 0 { 0 }`), documented as "correct, just not minimal".
  It is not correct: an all-negative block took `max <= 0 → bitsPerValue = 0`, which
  encodes every value in it as the constant `0`; a mixed-sign block truncated each
  negative value to the low `bitsPerValue` bits. The read side already handled
  negative minima correctly, so this was purely a writer defect.
- **Consequence**: silent data loss for any signed sequence. Latent today — every
  caller (`term_vectors`' prefix/suffix lengths, frequencies, positions, offsets,
  payload lengths, fields-per-doc) is non-negative — but `BlockPackedWriter` is a
  general signed-`long` writer and nothing in the signature said otherwise.
- **Fixed**: `encode_all` is now a full port of `flush`, min-value delta encoding
  included, so it is byte-identical to Java's for the same input rather than merely
  decodable. Tests: `encode_all_round_trips_negative_and_mixed_sign_values`
  (all-negative, mixed-sign, `i64::MIN`/`i64::MAX` in one block, all-`i64::MIN`,
  a 130-value two-block sequence straddling zero) and
  `encode_all_matches_javas_flush_min_value_choice` (pins the `bitsPerValue == 0`
  constant-block token and the `min = max(0, max - maxValue(bits))` lowering,
  neither of which a round-trip alone would catch).

### 7. `[CORRECTNESS]` the block `minValue` was read with the wrong varint

- **Java**: `BlockPackedReaderIterator.readVLong` is a *different* varint from
  `DataInput.readVLong` — at most 8 continuation groups (bits 0..55), then a ninth
  byte contributing all 8 of its bits at shift 56, top bit included. Its writer twin
  (`AbstractBlockPackedWriter.writeVLong`) has the matching `k++ < 8` cap. This exists
  precisely so a negative `minValue` round-trips.
- **We did**: `input.read_vlong()`, the generic Lucene varint, which treats the ninth
  byte's top bit as a continuation marker.
- **Consequence**: any block whose zigzag-encoded `minValue` needs bit 63 set
  (`|minValue|` around 2^62 and up) is mis-decoded, and the reader then eats a byte
  of the following block — desynchronising the rest of the stream, not just one value.
  Unreachable from today's term-vector data; reachable from any Java-written
  `BlockPackedWriter` stream.
- **Fixed**: local `read_min_value_vlong`/`write_min_value_vlong` ported from Java.
  Test: `min_value_vlong_round_trips_the_nine_byte_negative_capable_form`
  (`0`, `0x7f`, `0x80`, `2^56 - 1`, `2^56`, `u64::MAX`, and neighbours), asserting the
  9-byte cap as well as the round trip.

### Verdict

Swept clean.

---

## `crates/lucene-codecs/src/direct_reader.rs`

Java: `util/packed/DirectReader.java` (14 `DirectPackedReaderN` classes),
`util/packed/DirectWriter.java`.

| Rust `fn` | Java | Verdict |
|---|---|---|
| `get` | `DirectPackedReader1..64.get` collapsed into one formula | identical bit layout; validation divergent — finding 8 |
| `encode` | `DirectWriter.add`/`flush`/`encode` | identical output. Java flushes in `bufferSize`-value chunks rounded up to a multiple of 64; `64 * bitsPerValue` is a whole number of bytes for every width, so chunk boundaries are byte-aligned and the concatenation is the same bitstream this produces in one pass |
| `SUPPORTED_BITS` | `DirectWriter.SUPPORTED_BITS_PER_VALUE` | identical: `1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64` |
| `is_supported_bits` (new) | `DirectWriter.checkBitsPerValue` | added — finding 8 |
| `unsigned_bits_required` | `DirectWriter.unsignedBitsRequired` = `roundBits(PackedInts.unsignedBitsRequired)` | identical, including the `max_value == 0 → 1` floor |
| `padding_bytes_needed` | `DirectWriter.paddingBytesNeeded` | identical (`>32 → 64-b`, `>16 → 32-b`, `>8 → 16-b`, else 0) |
| — | `DirectWriter.bitsRequired(long)` (throws on negative), `DirectWriter.bytesRequired` | not ported, `INTENTIONAL`: callers use the unsigned form and compute lengths themselves |
| — | `DirectReader.getMergeInstance` | not ported — finding 9 |

### 8. `[CORRECTNESS]` `get` accepted unsupported widths and shifted by them

- **Java**: `DirectReader.getInstance`'s `switch` has a `default:` that throws
  `IllegalArgumentException("unsupported bitsPerValue")`, and `DirectWriter.getInstance`
  binary-searches `SUPPORTED_BITS_PER_VALUE` and throws.
- **We did**: `get` documented "the caller validates this at parse time" and computed
  `if bits_per_value == 64 { u64::MAX } else { (1u64 << bits_per_value) - 1 }`. For a
  width above 64 that is a shift overflow: a panic in debug, and in release a masked
  shift producing a **plausible wrong value with no error at all**.
- **Reachability**: `doc_values`'s non-varying entry parse does validate the width, and
  `direct_monotonic` now does (finding 11) — but `doc_values::read_varying_block` reads
  a per-block `bits_per_value` byte mid-lookup from the `.dvd` file with no check, and
  hands it straight to `get`. That is a real path from a corrupt file to a panic or a
  silent wrong value.
- **Fixed**: `get` rejects anything outside `SUPPORTED_BITS`. The check is a shift and a
  test against a `u64` bit set, not a scan of the table. Test:
  `unsupported_bit_width_is_rejected_not_a_shift_overflow` (both directions: every
  rejected width errors, every supported width still succeeds).
- **Cost, measured**: Java validates once per reader, this validates once per read.
  A first attempt used a `u128` bit set (needed because 64 is a legal width) and cost
  ~22% of `direct_monotonic/get_block`; splitting 64 out into its own comparison so the
  bit set fits in a `u64` brought it to roughly 5%, which the `direct_monotonic`
  restructuring below more than pays back.

### 9. `[PERF]` `DirectReader.getMergeInstance` is not ported

Java's merge instance buffers 128 values at a time with a width-specialised bulk
loop (`readLongs`/`readInts` at 8/2 values per load) and is "typically faster at
sequential access but slower at random access". This port has one random-access
formula. **Recorded, not fixed**: it is a pure speed specialisation for the merge
read pattern, it only pays off where a caller reads a long ascending run, and this
port has no merge-time doc-values reader that would hold one. Costed at the point
that reader exists.

### Verdict

Swept clean, with finding 9 recorded.

---

## `crates/lucene-codecs/src/direct_monotonic.rs`

Java: `util/packed/DirectMonotonicReader.java`, `util/packed/DirectMonotonicWriter.java`.

| Rust `fn` | Java | Verdict |
|---|---|---|
| `load_meta` | `DirectMonotonicReader.loadMeta` + `Meta(numValues, blockShift)` | identical field order/widths; validation and layout divergent — findings 11, 13 |
| `get` | `DirectMonotonicReader.get` | identical arithmetic (`avg` stays `f32`, so `avgs[block] * blockIndex` truncates exactly as Java's `float * long` does); overflow behaviour divergent — finding 12 |
| `write` | `DirectMonotonicWriter.flush` + `add`/`finish` | identical, including the `float` linear estimate, the OR-based `maxDelta`, and `DirectWriter`'s trailing padding bytes. Block-shift range unchecked — finding 14 |
| `floor_index` | — | not-in-Java (this port's own rightmost-`<=` helper for `.fdx` chunk lookup) |
| — | `DirectMonotonicReader.binarySearch` / `getBounds` | not ported — finding 10 |
| — | `Meta.SINGLE_ZERO_BLOCK` | not ported — finding 13 |

### 10. `[MISSING]` `binarySearch`/`getBounds` are not ported

Java's `binarySearch(from, to, key)` returns the index or `-1 - insertionPoint`, and
runs as many iterations as it can against cheap per-block bounds
(`min + avg*i` .. `+ (1 << bpv) - 1`) before touching the bit-packed reader, to avoid
page faults. **Recorded, not fixed**: no caller needs those semantics —
`floor_index` (rightmost `<= key`) is what stored fields' chunk lookup wants, it is
already tested, and the page-fault avoidance `getBounds` exists for does not apply to
an in-memory mmap slice this port has already faulted in. Noted in `docs/parity.md`.

### 11. `[CORRECTNESS]` a corrupt per-block `bitsPerValue` was not rejected

- **Java**: `DirectMonotonicReader.getInstance` eagerly constructs a `DirectReader`
  for every block with `bpvs[i] != 0`, so an unsupported width throws at load time.
- **We did**: `load_meta` pushed the raw byte and `get` handed it to
  `direct_reader::get` on first use — the shift overflow of finding 8, surfacing
  hundreds of lookups after the file that caused it.
- **Fixed**: `load_meta` rejects any non-zero `bpv` outside `SUPPORTED_BITS` with
  `Error::Corrupted`. Test: `unsupported_block_bits_per_value_is_rejected_at_load`.

### 12. `[CORRECTNESS]` `get` panicked instead of wrapping, and on an out-of-range index

- **Java**: `mins[block] + (long)(avgs[block]*blockIndex) + delta` is `long` arithmetic
  and wraps; an out-of-range `index` throws `ArrayIndexOutOfBoundsException`.
- **We did**: `+` (a debug-build overflow panic on corrupt metadata) and four raw
  index expressions (a slice-index panic).
- **Fixed**: `wrapping_add`, and the block lookup is a checked `get` returning
  `Error::Eof`. Tests: `get_wraps_like_java_instead_of_panicking_on_overflow`,
  `get_out_of_range_index_is_an_error_not_a_panic`.

### 13. `[PERF]` four parallel arrays, and an eager 32-byte error on the happy path

- **Java** keeps `long[] mins`, `float[] avgs`, `byte[] bpvs`, `long[] offsets` and
  indexes all four per `get` — a GC-era parallel-array shape the `rust-performance`
  skill names explicitly. It also collapses all-zero metadata to one shared
  `Meta.SINGLE_ZERO_BLOCK` (`blockShift = 63`, one block) purely to save heap.
- **We did**: transliterated the four `Vec`s — four bounds checks and up to four cache
  lines per lookup, where every `get` reads all four fields of exactly one block. And
  both slice lookups used `ok_or(Error::Eof { offset: 0 })`, which builds the 32-byte
  `lucene_store::Error` (it carries a `String`) **eagerly, on every successful lookup**.
- **Fixed**: `Meta` now holds one `Vec<Block>` of 24-byte structs, and both fallible
  lookups are `let ... else` (clippy rejects `ok_or_else` for a non-allocating value,
  and `let-else` avoids the eager construction without a closure).
- **Measured** (`direct_monotonic/get_block`, `benches/hot_paths.rs`, 16384 values,
  `blockShift = 10`), A/B on the same machine state by stashing only these two files:
  **HEAD 98.6–116.3 µs (3 runs) → 73.4–79.7 µs (3 runs), ≈ −25%**, *including* the
  per-read width validation of finding 8. Criterion's own `change:` line is not usable
  here — five sweep batches were compiling concurrently and machine state drifted
  between runs, which is why the comparison was done back-to-back against a stashed
  HEAD rather than against a saved baseline.
- `SINGLE_ZERO_BLOCK` **recorded, not fixed**: it is a heap optimisation with no
  behavioural effect (an all-zero `Meta` returns 0 for every index either way), and it
  costs us one `Vec` of 24-byte structs for a field whose addresses are all zero.

### 14. `[MISSING]` `write` does not enforce `MIN_BLOCK_SHIFT`/`MAX_BLOCK_SHIFT`

Java's `DirectMonotonicWriter` constructor throws for `blockShift` outside `[2, 22]`
and for `numValues < 0`. **Recorded, not fixed**: `block_shift` is never
caller-supplied at runtime here — every production call site passes a format constant
(doc values' `DIRECT_MONOTONIC_BLOCK_SHIFT = 16`, stored fields' 10) — and this
module's own tests deliberately use shifts of 0 and 1 to build small multi-block
fixtures, which the Java bound would forbid without buying any fidelity.

### Verdict

Swept clean, with findings 10, 13 (`SINGLE_ZERO_BLOCK`) and 14 recorded.

---

## `crates/lucene-codecs/src/indexed_disi.rs`

Java: `codecs/lucene90/IndexedDISI.java`.

Constants checked one by one against Java: `BLOCK_SIZE = 65536`,
`DENSE_BLOCK_LONGS = 1024`, `MAX_ARRAY_LENGTH = (1 << 12) - 1 = 4095`,
`DEFAULT_DENSE_RANK_POWER = 9`, `denseRankPower == -1` ⇔ our `NO_RANK = 0xFF`,
rank-table length `DENSE_BLOCK_LONGS >> (denseRankPower - 7)`, and the block-shape
thresholds (`cardinality <= 4095` SPARSE, `== 65536` ALL, else DENSE) — all identical.
Sentinel block: `NO_MORE_DOCS >>> 16 = 32767` with the single low value `65535`,
identical. Jump-table entry format (`int` index + `int` offset per block, `blockCount == 2`
suppressed) read and matched against `flushBlockJumps`.

| Rust `fn` | Java | Verdict |
|---|---|---|
| `dense_rank_bytes` (new) | `IndexedDISI` ctor / `writeBitSet` `denseRankPower` validation, `createRank`'s length | added — finding 15 |
| `decode_doc_ids` | `readBlockHeader` + the three `Method` bodies, driven to exhaustion | identical block decoding |
| `DisiCursor::advance_exact` | `IndexedDISI.advanceExact` | identical: walk blocks while `block < targetBlock`, then `block == targetBlock && advanceExactWithinBlock` |
| `DisiCursor::read_next_block_header` | `readBlockHeader` (+ `advanceBlock`'s iteration fallback) | identical, minus the jump table (finding 17) |
| `DisiCursor::ordinal_within_block` | `Method.SPARSE/DENSE/ALL.advanceExactWithinBlock` | identical ordinals. Verified against Java's `index`/`numberOfOnes`/`denseOrigoIndex`/`gap` bookkeeping: Java's `index` starts at `-1`, so `numberOfOnes = index + 1` is our `ordinal_base`, and `numberOfOnes - bitCount(word >>> target)` is our `ordinal_base + popcount(word & ((1 << bit) - 1)) + popcount(earlier words)` |
| `rank_of` | — | not-in-Java helper over `decode_doc_ids`' output |
| `write` | `writeBitSet` + `flush` | identical block shapes; no rank table, no jump table (findings 16, 17); contract check widened — finding 18 |
| — | `advance`, `nextDoc`, `intoBitSet`, `docIDRunEnd`, `index`, `cost`, `rankSkip`, `createBlockSlice`/`createJumpTable`, `asDocIndexIterator` | not ported, `INTENTIONAL` (documented in the module header: one-shot decode plus a forward-only cursor) |

### 15. `[CORRECTNESS]` an out-of-range `denseRankPower` underflowed the rank-table shift

- **Java**: both the `IndexedDISI` constructor and `writeBitSet` throw
  `IllegalArgumentException` unless `denseRankPower` is `-1` or in `7..=15`.
- **We did**: `if dense_rank_power != NO_RANK { DENSE_BLOCK_LONGS >> (dense_rank_power - 7) }`
  on a `u8`. For any power below 7 that is a subtraction underflow — a panic in debug,
  and in release a wrapped shift amount giving a silently wrong skip distance, which
  desynchronises the whole DISI region.
- **Reachability**: `dense_rank_power` is `input.read_byte()?` in both
  `doc_values::parse_*_entry` and `norms::parse_entry`, with no validation. It is also
  passed as a literal `0` by several in-tree callers (this crate's `hot_paths` bench,
  and test fixtures in `doc_values.rs` / `lucene-search/src/field_norms.rs`) — the bench
  builds ~9.4k present docs in block 0, i.e. a genuine DENSE block, and was running
  straight into the wrapped-shift path.
- **Fixed**: `dense_rank_bytes(power)` returns `0` for `0xFF`, the table size for
  `7..=15`, and `Error::Corrupted` otherwise; both `decode_doc_ids` and
  `DisiCursor::read_next_block_header` go through it. The bench now passes `0xFF`,
  which is what `indexed_disi::write` actually emits. Test:
  `dense_rank_power_outside_javas_legal_range_is_rejected` (rejects `0, 6, 16, 200` on
  both the decode and cursor paths; accepts `7, 9, 15, 0xFF` and still decodes the
  right doc through each).
  *Note for the coordinator*: the `dense_rank_power: 0` literals left in
  `doc_values.rs` and `lucene-search/src/field_norms.rs` test fixtures are still
  wrong data (0 is not a legal value for that metadata byte); they happen to build
  SPARSE-only regions so nothing reaches the DENSE branch. Left alone to avoid
  colliding with the batches that own those files.

### 16. `[MISSING]` `write` emits no DENSE rank table

Java's `writeBitSet` writes a rank table with `DEFAULT_DENSE_RANK_POWER = 9`
(`createRank`, one 2-byte entry per 512 docs) at the head of every DENSE block.
**Recorded, not fixed** — and it is not a compatibility defect: this port's writers
record `denseRankPower = 0xFF` in the matching metadata byte, which is Java's own
"no rank table" encoding, so a real Lucene reader constructed from our metadata
never looks for one. The cost is that `advance` inside a DENSE block of ours is a
word scan rather than a rank jump, for readers on either side. Already stated in
`indexed_disi::write`'s doc comment and `docs/parity.md`.

### 17. `[MISSING]` `write` emits no block jump table

Java appends an `(index, offset)` `int` pair per block after the last block and
returns the entry count for the metadata. **Recorded, not fixed**, same argument:
our writers record `jumpTableEntryCount = 0`, and `createBlockSlice`/`createJumpTable`
treat `<= 0` as "no jump table". Note the two sides coincide exactly for the
single-real-block case — Java itself suppresses the table when `blockCount == 2`
(one real block + `NO_MORE_DOCS`), so our output is byte-identical there.

### 18. `[CORRECTNESS]` `write`'s strictly-ascending check missed violations inside a block

`write` documented "panics if `doc_ids` isn't strictly ascending", but only compared
`doc_ids[i-1] < doc_ids[i]` at each block's *first* element; the inner loop that
consumes the rest of the block never checked. A descending or duplicated pair within
one 65536-doc range therefore produced a SPARSE block with out-of-order shorts, or a
DENSE block whose header cardinality counted a duplicate twice — corrupt output from
the exact input the contract says is rejected. **Fixed**: one `windows(2)` check over
the whole slice, same O(n) as before. Tests:
`write_rejects_a_descending_pair_inside_a_block`, `write_rejects_duplicate_doc_ids`.

### Verdict

Swept clean, with findings 16 and 17 recorded.

---

## Open items

1. **`for_encode` scratch reuse is available but unused** (finding 3). `ForUtil::encode`
   amortises the packing scratch across blocks the way Lucene's `ForUtil` instance does,
   but `postings_writer::write_full_block`/`write_full_position_block`/
   `write_full_offset_block` still call the one-shot free functions, so each block
   allocates and zeroes a fresh 1 KiB buffer. Threading a `&mut ForUtil` from the
   per-term loop through those three functions is the remaining step; it touches
   `postings_writer.rs`, which this batch does not own, and the benchmark could not
   resolve the win above the noise of a contended machine. `benches/for_util_decode.rs`
   now carries the `oneshot`/`reused` pair to settle it on a quiet machine.
2. **`doc_values::read_varying_block`** reads a per-block `bits_per_value` off disk with
   no validation (finding 8). It is now safe because `direct_reader::get` validates, but
   the natural place is the parse site, where Java puts it, and where it would cost
   nothing per read.
3. **`dense_rank_power: 0` literals** in `doc_values.rs` and
   `lucene-search/src/field_norms.rs` test fixtures (finding 15) — harmless today,
   invalid metadata, owned by other batches.
4. **Not ported, deliberately** and recorded above: `PForUtil.skip` (5),
   `DirectReader.getMergeInstance` (9), `DirectMonotonicReader.binarySearch`/`getBounds`
   (10), `Meta.SINGLE_ZERO_BLOCK` (13), `DirectMonotonicWriter`'s block-shift bounds
   (14), `IndexedDISI`'s rank (16) and jump (17) tables.

## Gate

`cargo fmt --all`, `cargo clippy -p lucene-codecs --all-targets -- -D warnings`,
`cargo test -p lucene-codecs`. All six swept files are clean. The working tree was
being edited concurrently by other M2 sweep batches throughout, so intermediate runs
failed to compile on `fst.rs`, `stored_fields.rs`, `terms_dict.rs`, `blocktree.rs` and
`norms.rs` at various points; those are not this batch's changes and the gate was
re-run until the tree was consistent.
