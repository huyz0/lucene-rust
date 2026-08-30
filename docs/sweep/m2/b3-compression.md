# M2 sweep — batch `b3-compression`

Files swept (every function in each):

- `crates/lucene-codecs/src/lz4.rs`
- `crates/lucene-codecs/src/deflate.rs`
- `crates/lucene-codecs/src/compound_format.rs`
- `crates/lucene-codecs/src/stored_fields.rs`
- `crates/lucene-codecs/src/lib.rs`

Java source of truth: `/home/tuong/work/lucene` @ 10.5.0. Confirmed with grep
that these formats have **not** moved to a newer package version — the pinned
codec (`Lucene104Codec`) still routes stored fields through
`org.apache.lucene.codecs.lucene90.Lucene90StoredFieldsFormat` and compound
files through `lucene90.Lucene90CompoundFormat`; only blocktree/postings moved
to `lucene103`/`lucene104`. `org.apache.lucene.codecs.compressing.CompressionMode`
is the live copy (there is a `backward_codecs` twin, out of scope).

## Summary

21 findings, numbered F1-F27 (numbering is per-batch and contiguous across
files; the gaps are section boundaries).

| Class | Count | Findings |
|---|---|---|
| `CORRECTNESS` | 4 | F9, F11, F21, F24 — two are code (a debug-build overflow panic on a corrupt `.cfe`; four hang/underflow/abort paths on corrupt `.fdt`/`.fdm` framing), two are doc comments that described behaviour the code does not have |
| `MISSING` | 4 | F3, F6, F19, F22 — `compressWithDictionary`, `HighCompressionHashTable` + the better-match loop, the BEST_SPEED writer's dictionary/sub-block framing, and every non-widest branch of `writeZFloat`/`writeZDouble`/`writeTLong` |
| `PERF` | 7 | F1, F4, F5, F8, F10, F20, F26 — five fixed and measured, two recorded open with reasons |
| `INTENTIONAL` | 6 | F2, F7, F12, F23, F25, F27 |

All `CORRECTNESS` and `MISSING` findings are fixed with tests. Open by choice:
F8 (DEFLATE encoder has no preset dictionary — `miniz_oxide` exposes none),
F10 (a fresh inflater per sub-block), F26's parenthetical (the writer API takes
a `&[Document]` slice rather than streaming), F27 (`document()` materializes a
whole `Document` rather than exposing a visitor).

---

## `crates/lucene-codecs/src/lz4.rs`

Java counterpart: `lucene/core/src/java/org/apache/lucene/util/compress/LZ4.java`
(whole class). `LowercaseAsciiCompression.java` is *not* this file's counterpart
— its decode half lives in `blocktree.rs::decompress_lowercase_ascii` (batch
b4), and its compress half is unported there.

### Method correspondence

| Java | Rust | Verdict |
|---|---|---|
| `LZ4.decompress` | `decompress` | identical semantics; match copy re-shaped (F4) |
| `LZ4.hash` | `hash` | identical |
| `LZ4.hashHC` | inlined as `hash(v, HASH_LOG_HC)` | identical |
| `LZ4.readInt` | `read4` | identical |
| `LZ4.commonBytes` (`Arrays.mismatch`) | `common_bytes` | identical result; word-at-a-time now (F5) |
| `LZ4.encodeLen` | `encode_len` | identical |
| `LZ4.encodeLiterals` | `encode_literals` | identical |
| `LZ4.encodeLastLiterals` | `encode_last_literals` | identical |
| `LZ4.encodeSequence` | `encode_sequence` | identical |
| `LZ4.HashTable` (abstract) | `trait HashTable` | ported |
| `LZ4.Table16` / `Table32` | folded into `FastCompressionHashTable`'s `Vec<u32>` | INTENTIONAL (F2) |
| `FastCompressionHashTable.reset` | `FastCompressionHashTable::reset` | ported (was **missing**: F1) |
| `FastCompressionHashTable.initDictionary` | `…::init_dictionary` | ported (was missing: F3) |
| `FastCompressionHashTable.get` | `…::get` | ported |
| `FastCompressionHashTable.previous` | `…::previous` (always `None`) | identical |
| `HighCompressionHashTable.*` (`reset`/`initDictionary`/`get`/`addHash`/`previous`) | `HighCompressionHashTable::*` | ported (was **missing**: F6) |
| `HashTable.assertReset` | — | not ported: assertion-only, no runtime behaviour |
| `LZ4.compress(bytes,off,len,out,ht)` | `compress_into` | ported |
| `LZ4.compressWithDictionary` | `compress_with_dictionary` | ported (was **missing**: F3) |
| — | `compress(&[u8]) -> Vec<u8>` | not-in-Java: one-shot convenience wrapper |

### Findings

1. **[PERF] Fixed — the match-finder hash table was 1 MiB, allocated and
   zero-filled on every `compress` call.** Java sizes it from the input
   (`hashLog = MEMORY_USAGE + 3 - bitsPerOffsetLog`, i.e. 13 below 64kB and 12
   above, always 16kB of memory) and, crucially, **reuses one table across
   calls without ever clearing it** — `get` re-verifies every candidate
   byte-for-byte, so stale entries are harmless. This port used a fixed
   `2^17`-slot `i64` table with a `-1` sentinel, freshly allocated per call:
   64x Java's memory and a full 1 MiB `memset` per compressed unit. With the
   new stored-fields framing that is **11 units per chunk**.
   Measured (release, this machine): allocating + touching the old
   `vec![-1i64; 1<<17]` costs **13.0 µs**; the new `vec![0u32; 1<<13]` costs
   **0.12 µs**, and in the steady state costs nothing at all because the table
   is reused. For scale, compressing a whole 64 kB block takes 71 µs, so on an
   8 kB sub-block the old table allocation was the dominant cost.
   Resolution: `FastCompressionHashTable` is now a reusable struct with Java's
   sizing and its "never clear" contract; `write_best_speed` allocates one per
   call and threads it through every unit of every chunk.
   Tests: `a_reused_fast_hash_table_stays_correct_across_many_inputs`,
   `fast_hash_table_switches_hash_log_above_64kb_and_still_round_trips`.
   Bench: `lz4/compress/{reused_table,fresh_table}` in
   `crates/lucene-codecs/benches/lz4_codec.rs` (71.1 µs vs 76.7 µs per 64 kB —
   the residual is the 32 kB allocation the wrapper still does).

2. **[INTENTIONAL] `Table16`/`Table32` collapsed to one `Vec<u32>`.** Java
   keeps two physical widths purely to hold its 16 kB budget while widening
   the stored offset above 64 kB inputs. A `u32` slot is never *less* precise
   than either (Java only picks `Table16` when offsets provably fit in 16
   bits), so the same candidate is produced at every step; only the memory
   doubles to 32 kB for the sub-64 kB case. The **hash log** is reproduced
   exactly, because that is what decides which offsets collide and therefore
   which matches get found.

3. **[MISSING] Fixed — `compressWithDictionary` (preset-dictionary
   compression) was not ported at all.** Only the zero-dictionary case
   existed, which is why the stored-fields writer emitted `dictLength = 0` and
   one block covering the whole chunk. Consequence: worse ratio than Lucene on
   every chunk, and — because a single whole-chunk block cannot be skipped —
   no way for a reader to fetch one document without inflating everything (see
   `stored_fields` F20). Resolution: full port including `initDictionary`.
   Test: `compress_with_dictionary_round_trips_against_the_preset_dictionary`
   (asserts the dictionary actually *helps*, not just that it round-trips),
   `high_compression_with_dictionary_round_trips`.

4. **[PERF] Fixed — match copying was a bounds-checked byte loop.** Java
   splits into `System.arraycopy` for the non-overlapping case and a byte loop
   only for the overlapping one. The port always used the byte loop, paying two
   bounds checks per output byte. Resolution: bulk `copy_within` in runs of at
   most `matchDec` bytes, which is correct for both cases (each run's source is
   wholly inside already-written bytes) and needs no branch to distinguish
   them. Measured by temporarily reverting: **103.4 µs → 56.9 µs per 64 kB of
   text (~2x)**, and 3.65 µs → 3.47 µs (~6%) on run-length data.
   Bench: `lz4/decompress/{text,run_length}`.
   Test: `overlapping_match_with_a_multi_byte_period_repeats_the_whole_run`
   (matchDec=3, matchLen=8 — a single memmove would be wrong here).

5. **[PERF] Fixed — `common_bytes` compared one byte at a time** where Java's
   `Arrays.mismatch` is a JIT intrinsic that compares a vector register at a
   time. Now an 8-byte `u64` loop with a byte tail; `u64::from_le_bytes` pins
   the interpretation so `trailing_zeros() / 8` names the first differing byte
   identically on either endianness. Folded into the compress timings above.
   Test: `common_bytes_counts_across_the_word_loop_boundary`.

6. **[MISSING] Fixed — `HighCompressionHashTable` and the "try to find a
   better match" loop were not ported.** The module doc claimed this was out of
   scope because nothing needed it. That was wrong for the wrong reason: it is
   true that `Lucene90StoredFieldsFormat` and `Lucene90TermVectorsFormat` both
   use `FastCompressionHashTable` (via `CompressionMode.FAST` and
   `LZ4WithPresetDictCompressionMode`), and `CompressionMode.FAST_DECOMPRESSION`
   is used by nothing in `lucene/core`. But
   `Lucene103BlockTreeTermsWriter.writeBlocks` constructs
   `new LZ4.HighCompressionHashTable()` to compress terms-block suffixes — a
   writer this port will need. Resolution: ported in full, including
   `MAX_ATTEMPTS = 256`, the `chainTable` delta encoding, and `reset`'s
   two branches (partial chain-table wipe after a sub-64 kB input, full wipe
   above). The better-match loop in `compress_with_dictionary` is now present
   and is a no-op for the fast table, exactly as in Java.
   Tests: `high_compression_hash_table_beats_the_fast_one_and_round_trips`
   (ratio 0.202 vs 0.455 for the fast table on the bench payload),
   `high_compression_hash_table_is_reusable_across_short_and_long_inputs`
   (both `reset` branches, alternating).

7. **[INTENTIONAL] `lz4` is now a `pub` module.** `LZ4` is a public class in
   Lucene, both hash tables are part of its contract, and — following the
   precedent `for_util`/`direct_reader` already set — a codec kernel this hot
   needs a benchmark that can call it from outside the crate.

### Verdict

Swept clean. The compressor is now a complete port rather than a scoped-down
one; nothing in `LZ4.java` is left unported except `assertReset`, which has no
runtime behaviour.

---

## `crates/lucene-codecs/src/deflate.rs`

Java counterpart:
`lucene/core/src/java/org/apache/lucene/codecs/lucene90/DeflateWithPresetDictCompressionMode.java`
(its `DeflateWithPresetDictDecompressor.doDecompress` / `DeflateWithPresetDictCompressor.doCompress`).
`compressing/CompressionMode.java`'s `DeflateCompressor`/`DeflateDecompressor`
(the non-preset-dict `HIGH_COMPRESSION` mode) is **not** reachable from
`Lucene90StoredFieldsFormat` and is deliberately not ported.

### Method correspondence

| Java | Rust | Verdict |
|---|---|---|
| `DeflateWithPresetDictCompressor.doCompress` (the deflate call itself) | `compress` | divergent: no preset dictionary (F8) |
| `DeflateWithPresetDictDecompressor.doDecompress` (the inflate call itself) | `decompress` | identical result |
| `Inflater.setDictionary(bytes, 0, dictLength)` | back-references into `dest[..d_off]` via `TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF` | equivalent |
| `decompress`'s framing/skip loop | lives in `stored_fields::decompress_unit` | see that file |

### Findings

8. **[PERF, recorded] The encoder does not use a preset dictionary.**
   `miniz_oxide` exposes no `deflateSetDictionary`, so every sub-block is a
   self-contained stream and compresses worse than Java's, which primes each
   sub-block's window with the chunk dictionary. This is a ratio divergence,
   not a correctness one: the wire format only requires that a sub-block's
   compressed bytes inflate into `buffer[dictLength..]`, and the *decoder* side
   does back-reference into the dictionary correctly (proven by the
   Java-written `stored_fields_best_compression_index` fixture). Not fixed:
   the fix is a new DEFLATE encoder or a different crate, not a contained
   change. Recorded here and in `docs/parity.md`.

9. **[CORRECTNESS — doc] Fixed: two stale claims in the module docs.** The
   header said the compressed-length vints are "grouped together before any of
   the actual compressed bytes" — that is LZ4's framing; DEFLATE interleaves
   them, which `decompress_unit`'s own (correct) comment already said. And the
   compression-level note claimed Lucene passes `Deflater.BEST_COMPRESSION`
   (level 9); `DeflateWithPresetDictCompressionMode.newCompressor()` passes
   **6**, which is what this port already used. Both corrected. (Doc drift is
   a bug per AGENTS.md; nothing about the emitted bytes changed.)

10. **[PERF, recorded] A fresh `DecompressorOxide` (~11 kB) per sub-block.**
    Java reuses one `Inflater` and one `compressed` scratch array for a whole
    chunk. Not fixed: it would mean leaking a `miniz_oxide` type through
    `decompress`'s signature, and the block-skipping fix below cut the number
    of these allocations per document read from "every sub-block in the chunk"
    to "the dictionary plus one sub-block" — which is the larger factor. Left
    open, recorded.

### Verdict

Swept clean for correctness. Two PERF items open by choice (F8, F10), both
recorded with their reasoning.

---

## `crates/lucene-codecs/src/compound_format.rs`

Java counterparts:
`lucene/core/src/java/org/apache/lucene/codecs/lucene90/Lucene90CompoundFormat.java`
and `Lucene90CompoundReader.java`.

### Method correspondence

| Java | Rust | Verdict |
|---|---|---|
| `Lucene90CompoundReader.readEntries` + `readMapping` | `parse_entries` | identical (incl. duplicate-id rejection) |
| `Lucene90CompoundReader` ctor's header/checksum/length checks | `check_data_header_footer` | identical; overflow-hardened (F11) |
| `Lucene90CompoundReader.openInput` | `open_input` | identical (`IndexInput.slice` bounds) |
| `Lucene90CompoundFormat.write` + `writeCompoundFile` | `write` | identical (size-ascending order, 64-byte alignment, header-id + footer-checksum verification) |
| `CodecUtil.verifyAndCopyIndexHeader` + footer re-derivation | `verify_sub_file` + verbatim copy | equivalent (the re-derived footer is byte-identical to the copied one) |
| `CodecUtil.indexHeaderLength(codec, "")` | `index_header_length` | identical |
| `Lucene90CompoundReader.listAll` | `CompoundEntries::names` | equivalent (no segment-name prefixing: this port strips prefixes throughout) |
| `Lucene90CompoundReader.fileLength` | `CompoundEntries::get(..).length` | equivalent |
| `Lucene90CompoundReader.close` / `getPendingDeletions` / `checkIntegrity` / `ensureOpen` | — | not ported: `Directory`-lifecycle surface this port does not expose (it hands back `SliceInput`s over already-mapped bytes, so there is nothing to close, and integrity is checked at `check_data_header_footer`) |

### Findings

11. **[CORRECTNESS] Fixed — `offset + length` on entries straight off a
    corrupt `.cfe` could panic.** The expected-length cross-check computed
    `(e.offset + e.length) as usize` in `i64`. Java's `long` addition silently
    wraps and then simply fails the comparison; Rust's panics on overflow in a
    debug build, i.e. a corrupt file could abort a debug-built process rather
    than returning an error. Now `saturating_add` + `usize::try_from`, so it
    lands in the existing `WrongLength` error either way.
    Tests: `overflowing_entry_extent_is_a_length_error_not_a_panic`,
    `negative_entry_offset_is_rejected_by_open_input`.

12. **[INTENTIONAL] Slice bounds.** A negative `.cfe` offset becomes a huge
    `u64` in `open_input`; `SliceInput::slice_input` rejects it via its
    `buf.get(start..end)` bound, matching Java's `IndexInput.slice`
    `IllegalArgumentException`. Now covered by a test rather than assumed.

### Verdict

Swept clean.

---

## `crates/lucene-codecs/src/stored_fields.rs`

Java counterparts:
- `codecs/lucene90/Lucene90StoredFieldsFormat.java` (mode → chunk geometry)
- `codecs/lucene90/LZ4WithPresetDictCompressionMode.java`
- `codecs/lucene90/DeflateWithPresetDictCompressionMode.java`
- `codecs/lucene90/compressing/Lucene90CompressingStoredFieldsReader.java`
- `codecs/lucene90/compressing/Lucene90CompressingStoredFieldsWriter.java`
- `codecs/lucene90/compressing/FieldsIndexReader.java` / `FieldsIndexWriter.java`
- `codecs/lucene90/compressing/StoredFieldsInts.java`

### Chunk geometry — checked constant by constant

| Constant | Java | Rust | |
|---|---|---|---|
| BEST_SPEED chunk size | `10 * 8 * 1024` | `BEST_SPEED_CHUNK_SIZE` | ✓ |
| BEST_SPEED maxDocsPerChunk | `1024` | `BEST_SPEED_MAX_DOCS_PER_CHUNK` | ✓ |
| BEST_COMPRESSION chunk size | `10 * 48 * 1024` | `BEST_COMPRESSION_CHUNK_SIZE` | ✓ |
| BEST_COMPRESSION maxDocsPerChunk | `4096` | `BEST_COMPRESSION_MAX_DOCS_PER_CHUNK` | ✓ |
| blockShift (both) | `10` | `INDEX_BLOCK_SHIFT` | ✓ |
| `NUM_SUB_BLOCKS` (both modes) | `10` | `NUM_SUB_BLOCKS` | ✓ (was missing for LZ4) |
| LZ4 `DICT_SIZE_FACTOR` | `2` | `LZ4_DICT_SIZE_FACTOR` | ✓ (was missing) |
| DEFLATE `DICT_SIZE_FACTOR` | `6` | `DEFLATE_DICT_SIZE_FACTOR` | ✓ |
| type bits | `TYPE_BITS = bitsRequired(5) = 3`, `TYPE_MASK = 7` | same | ✓ |
| type tags | `STRING 0, BYTE_ARR 1, NUMERIC_INT 2, FLOAT 3, LONG 4, DOUBLE 5` | same | ✓ |
| tlong scales | `SECOND 1000`, `HOUR`, `DAY`, `0x40/0x80/0xC0` | same | ✓ |
| versions | `VERSION_START/CURRENT = 1`, `META_VERSION_START = 0`, index `VERSION_START/CURRENT = 0` | same | ✓ |

### Method correspondence

| Java | Rust | Verdict |
|---|---|---|
| `Lucene90CompressingStoredFieldsReader` ctor + `FieldsIndexReader` ctor | `open` | identical checks, same order; mode sniffed from the data codec name instead of the `.si` attribute (INTENTIONAL, pre-existing) |
| `FieldsIndexReader.getBlockID` | `direct_monotonic::floor_index(.., 0, num_chunks, ..)` | equivalent — Java searches `[0, totalChunks+1)`, this searches `[0, totalChunks)`; the excluded entry is the `maxDoc` sentinel, which can never be the answer for `docID < maxDoc` |
| `FieldsIndexReader.getBlockStartPointer` | `block_start_pointer` | identical |
| `FieldsIndexReader.getBlockLength` | — | not ported: only used by Java's prefetch/`merging` paths |
| `BlockState.doReset` | inlined in `document` | identical (docBase/token/`chunkDocs>>>2`/sliced bit/`chunkDocs == 1` special case/length-vs-numStoredFields validation) |
| `BlockState.document` | `document` | now range-limited (F20) |
| `readField` | `read_field` | identical |
| `skipField` | — | not-in-Java-shaped: this port materializes whole `Document`s, there is no visitor to say "skip" |
| `readZFloat` / `readZDouble` / `readTLong` | `read_zfloat` / `read_zdouble` / `read_tlong` | identical |
| `DataInput.readZInt` | `read_zint` | identical |
| `StoredFieldsInts.readInts`(+`8/16/32`) | `read_bulk_ints` | identical incl. the transposed 128-value block layout |
| `StoredFieldsInts.writeInts`(+`8/16/32`) | `write_bulk_ints` | identical (`max` vs Java's OR-accumulate pick the same bucket, since the thresholds are all-ones masks) |
| `Lucene90CompressingStoredFieldsWriter.finishDocument`/`triggerFlush`/`flush` | `write_chunked` | identical trigger order and dirty accounting |
| `writeHeader` | inline in `write_chunked` | identical token packing (`(numBufferedDocs << 2) \| dirtyBit \| slicedBit`) |
| `saveInts` | `chunk_docs == 1` branch + `write_bulk_ints` | identical |
| `writeField(...)` × 7 overloads | `serialize_doc` + `write_field` | identical |
| `writeZFloat` / `writeZDouble` / `writeTLong` | `write_zfloat` / `write_zdouble` / `write_tlong` | ported exactly (were widest-form-only: F22) |
| `FieldsIndexWriter.writeIndex` / `finish` | `write_index_and_meta` | identical arrays and meta field order |
| `LZ4WithPresetDictCompressor.compress`/`doCompress` | `write_best_speed`'s unit closure | ported (was divergent: F19) |
| `DeflateWithPresetDictCompressor.compress`/`doCompress` | `write_best_compression`'s unit closure + `write_deflate_unit` | ported; encoder-side dictionary still absent (deflate F8) |
| `LZ4WithPresetDictDecompressor.readCompressedLengths`/`decompress` | `decompress_unit` (BestSpeed arm) | ported incl. block skipping (F20) |
| `DeflateWithPresetDictDecompressor.decompress`/`doDecompress` | `decompress_unit` (BestCompression arm) | ported incl. block skipping (F20) |
| `copyOneDoc` / `copyChunks` / `merge` / bulk-merge | — | not ported: segment merging is out of this port's current scope |
| `serializedDocument` / prefetch cache / `merging` mode | — | not ported: no streaming `StoredFieldVisitor` consumer exists |

### Findings

19. **[MISSING/PERF] Fixed — the BEST_SPEED writer emitted a degenerate
    single-block frame.** It wrote `dictLength = 0`, `blockLength = <whole
    chunk>`, an empty dictionary unit, and one LZ4 block covering everything.
    Java writes
    `dictLength = min(64kB, len / (NUM_SUB_BLOCKS * DICT_SIZE_FACTOR))` and
    `blockLength = ceil((len - dictLength) / NUM_SUB_BLOCKS)`, compresses the
    dictionary first and then each sub-block *against* it, and writes every
    unit's compressed length up front before any bytes. Consequences of the
    old shape: (a) worse ratio, since no cross-sub-block redundancy was
    exploited; (b) a reader could not skip anything — the whole chunk was one
    indivisible block. Resolution: the unit closure now reproduces Java's
    geometry exactly, using `lz4::compress_with_dictionary` and one reused
    `FastCompressionHashTable`. Re-verified against real Lucene 10.5.0
    (`scripts/verify-write-path.sh` → `VerifyStoredFields`, all four segments).

20. **[PERF] Fixed — reading one document decompressed the entire chunk.**
    `decompress_unit` took only `original_length` and always inflated
    everything; `document()` then sliced the requested range out. Java's
    `Decompressor.decompress(in, originalLength, offset, length, bytes)`
    contract is the opposite: it skips whole sub-blocks that do not intersect
    `[offset, offset+length)` by their recorded compressed length (which is the
    entire reason those lengths are on disk), and both preset-dict modes
    implement it. On a full 80 kB BEST_SPEED chunk that is a **~7x** reduction
    in bytes inflated per document (dictionary ≈ len/20 plus one of ten
    sub-blocks, vs all of it), and it grows with chunk size.
    Resolution: `decompress_unit` now takes `(offset, length)` and appends
    exactly that range, skipping non-intersecting sub-blocks; it always leaves
    the reader at the end of the unit so a `sliced` chunk's back-to-back units
    still line up. `document()` cuts the wanted range against each unit's own
    extent. The dictionary is still always inflated, as in Java, because the
    sub-blocks back-reference into it.
    Tests: `every_document_position_inside_a_multi_sub_block_chunk_reads_back`
    (400 docs, ~40 kB chunk — documents inside the dictionary, across the
    dictionary/first-sub-block seam, mid-chunk and in the last sub-block) and
    its DEFLATE twin (whose length vints are interleaved, a separate branch).
    The Java-written fixtures (`stored_fields_index`,
    `stored_fields_best_compression_index`) still read back identically, which
    is the real proof that the skip arithmetic matches Lucene's own framing.
    `writer_produced_sliced_chunk_round_trips_through_the_reader` covers the
    one place the two levels of splitting compose (a >160kB document forces
    the writer to mark the chunk `sliced` and split it into several
    `chunk_size` units, each of which then has its own dictionary and
    sub-blocks); it asserts the `sliced` token bit structurally rather than
    assuming the size arithmetic worked out.

21. **[CORRECTNESS] Fixed — corrupt framing could hang, underflow, or abort
    the process.** Four hardening gaps, all reachable from a corrupt
    `.fdt`/`.fdm` and all of which Java either throws on or cannot hit:
    - `blockLength == 0` with bytes still to cover made the block-counting
      loop `while total < original_length { total += block_length }` spin
      forever (Java divides by `blockLength` and throws
      `ArithmeticException`);
    - `dictLength > originalLength` underflowed `original_length - plain` in
      the block loop;
    - `blockLength > originalLength` (impossible for a real writer, since
      `blockLength = ceil((len - dictLength) / 10)`) fed
      `vec![0u8; dictLength + blockLength]`, which *aborts* the process rather
      than erroring — same class of defect as the one batch b4 fixed in
      `blocktree.rs`; likewise `Vec::with_capacity(num_blocks)` for the
      compressed-length array, now a plain `Vec::new()` so a corrupt count
      hits EOF on the first `read_vint` instead of reserving gigabytes;
    - `.fdm`'s `chunkSize` was never validated, and a `sliced` chunk is
      divided by it — zero would make the reader's unit loop make no progress.
    All are now `Corrupted`. Tests:
    `zero_block_length_in_a_unit_header_is_rejected_not_hung_on`,
    `dictionary_longer_than_the_unit_is_rejected`,
    `zero_chunk_size_in_meta_is_rejected`.

22. **[MISSING] Fixed — `writeZFloat`/`writeZDouble`/`writeTLong` only ever
    emitted the widest encoding.** Every Java branch except the last was
    unported: the one-byte small-integer forms (`-1..=125` for float,
    `-1..=124` for double, both excluding negative zero), the five-byte
    `0xFE` "double that is exactly an f32" form, the four/eight-byte positive
    forms, and — the one that actually matters for real data — `writeTLong`'s
    second/hour/day scale headers, which are the whole point of that encoding
    (a 1-day timestamp went from 10 bytes to 1). The reader already handled
    all of them, so this was output size only, never a read failure; but the
    branches were reachable by any caller and unported.
    Resolution: ported exactly, including `Float.floatToIntBits`'s NaN
    canonicalization (Rust's `to_bits` preserves NaN payloads, Java's does
    not) and the `-0f` exclusion.
    Tests: `write_zfloat_uses_the_shortest_encoding_per_branch`,
    `write_zdouble_uses_the_shortest_encoding_per_branch`,
    `write_tlong_uses_the_scale_headers_to_shrink_timestamps` (all assert
    encoded *sizes*, not just round-trips — a round-trip test passes equally
    well against the widest-form encoder this replaced), plus round-trip tests
    covering every branch and NaN/−0.0.

23. **[INTENTIONAL] `write_deflate_unit` now emits Java's bare `vInt(0)` for a
    zero-length dictionary.** Chunks under 60 bytes have `dictLength == 0`, and
    the port previously compressed the empty input into a real 2-byte DEFLATE
    stream. This was investigated as a possible corruption bug — Java's
    reader calls `Inflater.inflate` with a zero-length output buffer on that
    path — and **deliberately falsified**: a new fixture segment (`_3`, two
    tiny documents) was written both ways and real Lucene 10.5.0 read both.
    Kept anyway for framing fidelity and two fewer bytes, and segment `_3` is
    kept because no previous BEST_COMPRESSION fixture was small enough to
    reach a zero-length dictionary at all.

24. **[CORRECTNESS — doc] Fixed: `read_bulk_ints`'s doc comment said it was
    ported "**without** Java's bit-transposed 128-value block layout … as a
    plain per-value loop".** The code does implement the transposed layout
    (and must, or it would misread every chunk of ≥128 documents). Rewritten
    to describe what it does.

25. **[INTENTIONAL] Extra `.fdt` length check.** `open` requires
    `fdt.len() == maxPointer + footerLength`, which Java only does for `.cfs`.
    It is exact for any well-formed file (the writer's footer starts at
    `maxPointer`) and catches truncation cheaply. Pre-existing; confirmed
    correct against `FieldsIndexWriter.finish`, kept.

26. **[PERF] Fixed — the writer serialized every document up front.**
    `write_chunked` began with
    `let payloads: Vec<Vec<u8>> = docs.iter().map(serialize_doc).collect()`,
    holding a whole segment's serialized payloads in memory at once (plus one
    `Vec` allocation per document, plus a `concat()` copy per chunk). Java's
    writer buffers exactly one chunk (`bufferedDocs` + `endOffsets` +
    `numStoredFields`, reset in `flush`). Now the same: one reused
    `chunk_buf`/`lengths`/`num_stored_fields` triple, documents serialized
    straight into it, and the chunk written from that buffer with no concat.
    Peak extra memory goes from O(segment) to O(chunkSize) and the per-document
    allocation disappears. (The public API still takes a `&[Document]` slice,
    so the caller's own documents are still all in memory — turning that into
    a streaming `add_document` writer is an API design task, not this batch's;
    recorded as still open.)

27. **[INTENTIONAL] `document()` materializes a whole `Document`** rather than
    exposing Java's `StoredFieldVisitor` streaming/`skipField` path or its
    `merging` bulk mode. Pre-existing scope decision; unchanged. Note this is
    now the *only* remaining "decodes more than asked" divergence, and it is a
    per-document one (all of one document's fields), not a per-chunk one.

### Verdict

Swept clean. Open by choice: the DEFLATE encoder's missing preset dictionary
(deflate F8), the whole-`Document` materialization (F27), and the non-streaming
`&[Document]` writer API (F26's parenthetical).

---

## `crates/lucene-codecs/src/lib.rs`

No Java counterpart — crate root, module declarations only. Changed in this
batch only to make `lz4` public (lz4 F7), with the rationale recorded inline
next to the existing `for_util`/`direct_reader` precedent.

### Verdict

Swept clean.

---

## Gate

`cargo fmt --all`, `cargo clippy -p lucene-codecs --all-targets -- -D warnings`,
`cargo test -p lucene-codecs`, plus `scripts/verify-write-path.sh`'s
`VerifyStoredFields` case (Rust writes, real Lucene 10.5.0 reads) for the
write-path changes.
