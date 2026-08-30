# m2 sweep — `b6-docvalues`

Java source of truth: `/home/tuong/work/lucene` (Lucene 10.5.0).

Files swept:

- `crates/lucene-codecs/src/doc_values.rs`
- `crates/lucene-codecs/src/doc_values_updates.rs`
- `crates/lucene-codecs/src/norms.rs`
- `crates/lucene-codecs/src/live_docs.rs`

---

## `crates/lucene-codecs/src/doc_values.rs`

Java counterparts (all under `lucene/core/src/java/org/apache/lucene/`):

- `codecs/lucene90/Lucene90DocValuesFormat.java`
- `codecs/lucene90/Lucene90DocValuesProducer.java`
- `codecs/lucene90/Lucene90DocValuesConsumer.java`
- `index/DocValuesSkipper.java`
- `codecs/DocValuesProducer.java`, `codecs/DocValuesConsumer.java` (abstract
  bases; no Rust counterpart by design — this port has no producer/consumer
  class hierarchy, see the `architecture` skill's "port by on-disk format")

### Method correspondence

Read side (`Lucene90DocValuesProducer`):

| Rust | Java | Verdict |
|---|---|---|
| `parse_meta` | ctor + `readFields` | divergent → **fixed** (F1); no meta-vs-data-vs-skipIndex version cross-check (see #10) |
| `read_numeric_entry` | `readNumeric(IndexInput, NumericEntry)` | identical (plus a stricter `bitsPerValue`/`blockShift` validation Java defers to `DirectReader.getInstance`) |
| `read_binary_entry` | `readBinary` | identical |
| `read_sorted_entry` | `readSorted` | identical |
| `read_sorted_numeric_entry` | `readSortedNumeric` | identical |
| `read_sorted_set_entry` | `readSortedSet` | identical |
| `read_skipper_meta` | `readDocValueSkipperMeta` | identical |
| `infer_max_value_counts` | `inferMaxValueCounts` | **added** (F1) |
| `check_data_header_footer` | ctor's data-file `checkIndexHeader` + `retrieveChecksum` | identical |
| `check_skip_index_header_footer` | ctor's skip-index `checkIndexHeader` | divergent → **fixed** (F2) |
| `parse_skip_index` | *(no Java equivalent — Java reads intervals lazily inside `getSkipper.advance`; this is `writeLevels`'s inverse)* | not-in-Java, intentional |
| `DocValuesSkipper` (`advance`, `advance_range`, `num_levels`, `min_doc_id`, `max_doc_id`, `min_value`, `max_value`, `doc_count`, `global_*`, `max_value_count`) | `getSkipper`'s anonymous `DocValuesSkipper` + `DocValuesSkipper.advance(long,long)` | **added** (F3) |
| `numeric_value` / `NumericReader::value` | `getNumeric` + `DenseNumericDocValues`/`SparseNumericDocValues` + `getNumericValues` | identical result; see #9 (PERF) |
| `decode_value` | `getNumericValues`'s table / gcd / min / raw branches | identical (branch order differs, outcome does not: Java tests `bitsPerValue == 0` before `blockShift`, but `doBlocks` always writes `0xFF`, so the two orders never disagree) |
| `read_varying_block` / `decode_value_varying_bpv` / `NumericReader::decode_varying` | `VaryingBPVReader.getLongValue` | identical, plus the cached-block state Java has |
| `binary_value` | `getBinary`'s four (dense/sparse × fixed/variable) branches | identical |
| `sorted_ord` | `getSorted().ordValue()` | identical |
| `sorted_numeric_values` | `getSortedNumeric`'s two branches | divergent (missing dense bound) → **fixed** (F4) |
| — | `getMergeInstance`, `close`, `checkIntegrity`, `TermsDict` (`next`/`seekExact`/`seekCeil`/`lookupTerm`), `rangeIntoBitSet`/`longValues`/`binaryValues`/`intoBitSet`/`docIDRunEnd`/`ordinalRangeIntoBitSet` bulk+SIMD APIs | **no Rust counterpart.** `TermsDict` lives in `crate::terms_dict` (a different sweep batch). The bulk/SIMD and `intoBitSet` families are iterator-shaped optimizations over an API this port does not have (it exposes random-access value lookups, not `DocIdSetIterator`s); recorded, not in this batch's scope. `checkIntegrity` is `codec_util::checksum_entire_file`'s job at the caller. |

Write side (`Lucene90DocValuesConsumer`):

| Rust | Java | Verdict |
|---|---|---|
| `write_dense_fields` + the five `write_single_dense_*` wrappers | `addNumericField`/`addBinaryField`/`addSortedField`/`addSortedNumericField`/`addSortedSetField` | identical shapes; address block shift divergent → **fixed** (F5) |
| the four `write_single_sparse_*` + `write_sparse_numeric_entry_body` | the same five methods' `numDocsWithValue != maxDoc` branch | identical |
| `write_numeric_values_body` | `writeValues` + `writeValuesSingleBlock` | divergent in the constant case → **fixed** (F6); `doBlocks` still unimplemented (#8) |
| `compute_gcd` / `gcd_i64` | `writeValues`'s gcd loop / `MathUtil.gcd` | identical (incl. the `[MIN/2, MAX/2]` overflow guard) |
| `write_terms_dict` | `addTermsDict` + `writeTermsIndex` + `compressAndGetTermsDictBlockLength` | identical layout; block shift fixed (F5); LZ4 preset dictionary intentionally absent (#8b) — now proven readable by real Lucene (F7) |
| `common_prefix_len` | `StringHelper.bytesDifference` / `sortKeyLength` | identical for the sorted-unique inputs the callers guarantee |
| `build_sorted_dict_and_ords` | `DocValues.singleton(...)` + `SortedSetSelector.wrap` plumbing | equivalent |
| `finish_field_list_and_footers` / `new_meta_output` / `new_data_output` | ctor + `close` | identical |
| — | `writeValuesMultipleBlocks` / `writeBlock`, `writeSkipIndex` / `writeLevels` / `buildLevel` / `getLevels` / `SkipAccumulator`, `isSingleValued`, `maybeGrowBuffer` | **no Rust counterpart** — see #8 (varying-bpv blocks) and #8c (skip-index write). `isSingleValued`/`maybeGrowBuffer` are Java-shape helpers with no Rust analogue needed. |

### Findings

1. **[MISSING] `inferMaxValueCounts` was not ported.** Java: for a `.dvm`
   older than `VERSION_SKIPPER_MAX_VALUE_COUNT` (2), `maxValueCount` is not on
   disk, and the producer runs a post-pass filling in `1` for any skipper on a
   field that is single-valued — NUMERIC/SORTED by construction, or
   SORTED_NUMERIC/SORTED_SET whose `numValues == numDocsWithField`. Us: every
   such entry stayed at the `-1` "unknown" sentinel, so a caller reading an
   older segment could never take a single-valued fast path.
   **Fixed** — `infer_max_value_counts` in `doc_values.rs`, called from
   `parse_meta` when `header.version < VERSION_SKIPPER_MAX_VALUE_COUNT`.
   Tests: `pre_version_2_numeric_skipper_max_value_count_is_inferred_as_one`,
   `current_version_skipper_max_value_count_is_read_not_inferred`,
   `pre_version_2_multi_valued_sorted_numeric_skipper_stays_unknown`.

2. **[CORRECTNESS] `.dvs` accepted a `VERSION_START` header.** Java opens the
   skip-index file with `checkIndexHeader(skipIn, ..., VERSION_SKIPPER_SEPARATE_FILE,
   VERSION_CURRENT, ...)` — a `.dvs` file does not exist before version 1
   (the skip data lived inline in `.dvd`). Us: `VERSION_START` as the minimum,
   so a corrupt/hand-forged version-0 `.dvs` was accepted and its bytes
   decoded as intervals. **Fixed** — `check_skip_index_header_footer` now
   passes `VERSION_SKIPPER_SEPARATE_FILE`. Test:
   `parse_skip_index_rejects_a_version_zero_dvs`.

3. **[MISSING] No `DocValuesSkipper` — `parse_skip_index` decoded the
   intervals but nothing could *skip* with them.** Java: `getSkipper` returns a
   cursor with `advance(int)`, `numLevels()`, per-level `minDocID`/`maxDocID`/
   `minValue`/`maxValue`/`docCount`, the global accessors, and the final
   `advance(long minValue, long maxValue)`. Three behaviours in `advance` are
   easy to get wrong and were reproduced deliberately:
   - the per-level arrays **persist across intervals** (a plain level-0
     interval leaves levels 1..3 holding the last interval that carried them,
     which is exactly what makes the trailing
     `while (levels < MAX && maxDocID[levels] >= target) levels++` widening
     meaningful);
   - a level's `maxDocID` is **stored before it is tested**
     (`if ((maxDocID[level] = input.readInt()) < target)`), so even the level
     whose bound failed leaves its value behind for that widening loop;
   - bailing out at level `L` **jumps the whole `8^L`-interval subtree**.
     Java's `SKIP_INDEX_JUMP_LENGTH_PER_LEVEL[L]` byte jump is exactly that
     subtree's byte size — verified by hand against `writeLevels`'s layout for
     all three levels — which over this port's already-decoded
     `Vec<SkipIndexInterval>` is a `cursor += 8^L`.

   **Fixed** — `pub struct DocValuesSkipper<'a>` + `pub const NO_MORE_DOCS`.
   Tests: nine unit tests (`skipper_*`), including one whose covered intervals
   carry an impossible `max_doc_id` so a linear walk instead of a subtree jump
   fails, plus two differential tests over the real 36,000-doc `.dvs` fixture
   (`skipper_walks_real_skip_index_and_brackets_the_real_values` asserts every
   interval's own `.dvd` values fall inside its level-0 min/max with an exact
   `docCount` match and that all 9 intervals are visited once;
   `skipper_advance_range_lands_on_the_first_intersecting_interval`).

4. **[CORRECTNESS] `sorted_numeric_values` had no dense upper bound.** Java's
   iterator is bounded by `maxDoc`. Us: the dense branch used `doc` as the rank
   with no check, so on a constant-encoded field (`bitsPerValue == 0`, where
   `decode_value` never indexes anything) *any* doc id returned the constant
   instead of an error — a silent wrong answer, not a decode failure. A dense
   entry means `numDocsWithField == maxDoc`, so that field is the bound.
   **Fixed** — `sorted_numeric_values` returns `Error::DocOutOfRange`. Test:
   `sorted_numeric_dense_doc_past_max_doc_rejected`.

5. **[PERF] Address arrays were written with `DirectMonotonic` block shift 0,
   not Java's 16.** Java uses `DIRECT_MONOTONIC_BLOCK_SHIFT = 16` for BINARY's
   variable-length end offsets, SORTED_NUMERIC/SORTED_SET's per-doc value
   ranges, and the terms dict's block and reverse-index addresses. Us: 0 — one
   block per value. Same values decode either way (the shift is stored per
   array), but the *metadata* then grows linearly with the value count: 21
   bytes of `.dvm` and one eagerly materialised `direct_monotonic::Meta` block
   per address. A 1M-doc variable-length BINARY field produced roughly 21 MB of
   `.dvm` and ~32 MB of heap on open, against Java's ~336 bytes and 16 blocks —
   and `.dvm` is fully parsed on every segment open. **Fixed** — one
   `DIRECT_MONOTONIC_BLOCK_SHIFT = 16` constant, used by all six call sites.
   Test: `direct_monotonic_block_shift_keeps_meta_size_constant` (a 100-doc and
   a 1000-doc field must produce the same `.dvm` size, and the 1000-doc field
   still round-trips value-for-value). Real Lucene reads the new bytes —
   `scripts/verify-write-path.sh` 14/14.

6. **[CORRECTNESS, byte-level] The constant encoding wrote a different `gcd`
   and `min` than Java.** Java runs the GCD scan *before* the `min >= max`
   branch and writes whatever it produced — `0` when every value is identical
   (every `v - firstValue` is 0), `1` when the `[MIN/2, MAX/2]` overflow guard
   tripped — and writes `MinMaxTracker`'s untouched `Long.MAX_VALUE` as `min`
   for an empty value array. Us: a hardcoded `gcd = 1` and `min = 0`. Neither
   field is read back (`bitsPerValue == 0` short-circuits in
   `getNumericValues`, and an empty field carries the `-2` docs-with-field
   marker so no doc ever resolves), so this was not observable — but it made
   the writer non-byte-identical to Java for the single most common encoding.
   **Fixed** — GCD hoisted into `compute_gcd`, called before the branch. Tests:
   `write_single_dense_numeric_field_all_equal_values_uses_constant_encoding`
   now asserts `gcd == 0`, and
   `write_single_dense_numeric_field_all_equal_extreme_values_record_gcd_one`
   covers the guard-tripped `gcd == 1` case.

7. **[MISSING, test coverage] The terms dictionary had never been read by real
   Lucene at a size where its structure exists.** The write-path fixture's only
   SORTED/SORTED_SET dictionaries were 3–5 terms: one LZ4 block, one empty
   reverse-index sort key. So `addTermsDict`'s flush-the-previous-block path,
   `writeTermsIndex`'s real `StringHelper.sortKeyLength` sort keys, and both
   prefix/suffix escape thresholds (`prefixLength >= 15`, `suffixLength >= 16`)
   were only ever validated against this port's own reader — precisely the
   shared-misreading blind spot the `differential-testing` skill warns about.
   **Fixed** — new fixture segment `_10`, a 1500-term dictionary (24 LZ4 blocks,
   one crossing of `TERMS_DICT_REVERSE_INDEX_SHIFT`, every adjacent pair past
   both escape thresholds), and `VerifyDocValues.java` now also round-trips
   **every** distinct term through real `SortedDocValues.lookupTerm` — the only
   API that exercises the reverse index and `seekBlock`, which `lookupOrd`
   never touches — plus `getValueCount` and a negative-insertion-point check
   for an absent term. Passes: `_10: all 1500 doc values verified against real
   Lucene`. This is also the first real-Lucene confirmation that this port's
   LZ4 (no preset-dictionary compressor: it emits a plain block where Java
   emits one compressed against the block's first term) is accepted by
   `TermsDict.decompressBlock`.

8. **[INTENTIONAL] Write-side scope cuts that remain, all pre-existing and
   documented:**
   - (a) `writeValuesMultipleBlocks`/`writeBlock` — the `doBlocks`
     varying-bits-per-value split (`minMax.spaceInBits > 0 &&
     blockMinMax.spaceInBits / minMax.spaceInBits <= 0.9`). The **read** side
     is complete and fixture-verified; the writer always emits one uniform
     width. Output stays valid and Lucene-readable, only larger for value sets
     with wildly varying per-block ranges.
   - (b) LZ4 preset dictionary in `addTermsDict`. Java compresses each block
     against its first term as a dictionary; this port compresses the block
     body alone. Java's decompressor only ever *allows* matches into the
     dictionary region, so both are legal — now proven, see #7. Costs
     compression ratio, not correctness.
   - (c) `writeSkipIndex`/`writeLevels`/`SkipAccumulator` — skip-index *write*.
     Deferred as previously recorded: nothing in this port requests a skip
     index at write time, so the serializer would be dead code.

9. **[PERF, not fixed — cross-batch dependency] `NumericReader` holds a decoded
   `Vec<i32>` of every present doc for a sparse field, where Java's
   `IndexedDISI` is O(1) memory.** 4 bytes per present doc (4 MB at 1M docs),
   plus an O(cardinality) decode at construction, in exchange for O(log N)
   *random* access — which Java's forward-only `advanceExact` cannot do at all.
   The obvious fix (back it with `indexed_disi::DisiCursor`) is **not** an
   improvement as `indexed_disi.rs` stands today, for two reasons found while
   reading it:
   - `DisiCursor::ordinal_within_block` recomputes the rank from the block
     start on **every** call (up to 1024 `i64` reads + popcounts inside a DENSE
     block), where Java's `IndexedDISI` carries incremental `index`/`word`
     state across advances. A cursor-backed reader would likely be *slower*
     than the binary search for a forward scan.
   - `DisiCursor::advance_exact` silently returns `Ok(None)` for a backward
     doc rather than erroring, so a cursor-backed `NumericReader` would answer
     "no value" for a random-order caller — a correctness hazard, not just a
     perf one.

   Making the cursor carry incremental rank state is the prerequisite, and
   `indexed_disi.rs` belongs to another sweep batch. **Recorded, not fixed.**

10. **[MISSING, minor] No meta/data/`.dvs` version cross-check.** Java's ctor
    throws `CorruptIndexException("Format versions mismatch: meta=..., data=...")`
    when the three files disagree. Us: `parse_meta`,
    `check_data_header_footer` and `check_skip_index_header_footer` each return
    their version and no caller compares them. Consequence is a missed
    corruption diagnostic on a mismatched file set (each file individually
    validates its own header, id and suffix, so a mismatched *segment* is
    already rejected); no wrong value can result. **Recorded** — the fix
    belongs at the call sites in `lucene-search`/`lucene-index`, not in this
    file.

### Verdict

Swept; findings 1–7 fixed with tests. Open: #9 (blocked on `indexed_disi.rs`,
another batch), #10 (caller-side), and the three intentional write-side scope
cuts in #8.

---

## `crates/lucene-codecs/src/norms.rs`

Java counterparts: `codecs/lucene90/Lucene90NormsFormat.java`,
`Lucene90NormsProducer.java`, `Lucene90NormsConsumer.java`;
`codecs/NormsProducer.java`/`NormsConsumer.java` (abstract bases, no Rust
counterpart by design).

### Method correspondence

| Rust | Java | Verdict |
|---|---|---|
| `parse_meta` | ctor + `readFields` | identical byte-for-byte, incl. the `0/1/2/4/8` `bytesPerNorm` rejection; does not validate field numbers against `FieldInfos` (#4) |
| `check_data_header_footer` | ctor's data-file `checkIndexHeader` + `retrieveChecksum` | identical |
| `norm_value` | `getNorms` + `DenseNormsIterator`/`SparseNormsIterator` | identical, all of const/1/2/4/8 and dense/sparse/empty |
| `read_value_at_ordinal` | the `switch (entry.bytesPerNorm)` in both iterators | identical (sign-extending `readByte`/`readShort`/`readInt`/`readLong`) |
| `num_bytes_per_value` | `numBytesPerValue` | **added** (F1) |
| `write_fields` | `addNormsField` (all branches) + `writeValues` + `close` | **added** (F1/F2/F3) |
| `write_single_dense_field` | `addNormsField`'s `numDocsWithValue == maxDoc` branch | now a wrapper over `write_fields` |
| `write_single_sparse_field` | `addNormsField`'s `else` branch | **added** (F2) |
| — | `getMergeInstance`, `close`, `checkIntegrity`, the `disiInputs`/`dataInputs`/`disiJumpTables` merge-instance caches | **no Rust counterpart** — merge-instance caching is a Java-object-lifecycle optimization with no analogue here (this port re-slices the already-in-memory `.nvd`) |

### Findings

1. **[MISSING] Only the constant and 1-byte-per-doc widths were written.** Java's
   `numBytesPerValue` picks the narrowest of `0/1/2/4/8`. Us: a value range
   outside `i8` returned `WriteError::RangeTooWide`, so a caller with a custom
   `Similarity` (Lucene's `computeNorm` returns a `long`, only BM25 happens to
   fit in a byte) simply could not write norms. **Fixed** — `num_bytes_per_value`
   ports the full ladder; `RangeTooWide` is gone. Test:
   `write_single_dense_field_uses_every_per_value_width` (all five widths, both
   signs, round-tripped through this port's reader) **plus** real-Lucene
   verification of the 2/4/8-byte segments (`_2`/`_3`/`_4`, below).

2. **[MISSING] No sparse norms writer.** Java's `addNormsField` writes an
   `IndexedDISI` docs-with-field structure whenever `numDocsWithValue !=
   maxDoc`, with the value array indexed by rank among present docs. Us: the
   dense marker only. **Fixed** — `write_single_sparse_field`. Two shapes
   degenerate on purpose, branching on the *count* exactly like Java rather
   than on how the caller phrased it: an empty list writes the `-2` "no
   document has this field" marker, and a list covering every doc writes the
   `-1` dense marker. Tests:
   `write_single_sparse_field_round_trips_and_skips_absent_docs`,
   `write_single_sparse_field_degenerates_to_the_dense_and_empty_markers`,
   `write_single_sparse_field_rejects_duplicate_and_out_of_range_doc_ids`,
   plus real-Lucene segment `_5`.

3. **[MISSING] One field per `.nvm`/`.nvd` pair.** Java's consumer writes every
   normed field of a segment into the same pair, one `addNormsField` call each
   — which is what any real multi-normed-field segment looks like. **Fixed** —
   `write_fields(&[NormsField], max_doc, ...)`, with the two single-field
   functions as wrappers. Tests:
   `write_fields_interleaves_several_fields_in_one_pair` (dense wide + sparse +
   constant, checking no cross-contamination and per-field width choice),
   `write_fields_rejects_an_empty_or_duplicated_field_list`, plus real-Lucene
   segment `_6`.

   All three fixes are verified in the write→read direction:
   `examples/write_norms_fixture.rs` now writes seven segments (`_0` 1-byte,
   `_1` constant, `_2` 2-byte, `_3` 4-byte, `_4` 8-byte, `_5` sparse with a
   leading/interior/trailing gap, `_6` three interleaved fields), and
   `fixtures/src/VerifyNorms.java` iterates each field through real
   `NumericDocValues`, asserting that docs the manifest marks absent are
   genuinely skipped. `scripts/verify-write-path.sh`: 14/14.

4. **[MISSING, not fixed] `parse_meta` does not validate field numbers against
   `FieldInfos`.** Java's `readFields` throws `CorruptIndexException` both for
   an unknown field number and for a field whose `FieldInfo` says it has no
   norms. Us: `parse_meta` takes no `FieldInfos` at all, so a corrupt `.nvm`
   naming a nonexistent field is accepted into `Norms.entries` and then simply
   never matched by an `entry(field_number)` lookup. Consequence is a missed
   corruption diagnostic; **no wrong value can result**, since every caller
   looks up by a field number it got from its own `FieldInfos`.
   **Recorded, not fixed**: threading `FieldInfos` through would touch 21 call
   sites across `lucene-codecs`, `lucene-search`, `lucene-index` and
   `lucene-ffi`, several of which are held by concurrent sweep batches. Worth a
   follow-up task once the sweep settles. (`doc_values::parse_meta` already
   takes `FieldInfos` and does validate.)

### Verdict

Swept; findings 1–3 fixed with tests and real-Lucene verification. Open: #4
(cross-crate signature change, deferred).

---

## `crates/lucene-codecs/src/live_docs.rs`

Java counterpart: `codecs/lucene90/Lucene90LiveDocsFormat.java`.

### Method correspondence

| Rust | Java | Verdict |
|---|---|---|
| `parse` | `readLiveDocs(Directory, SegmentCommitInfo, IOContext)` + `readLiveDocs(IndexInput, ...)` + `readFixedBitSet` | divergent (ghost bits) → **fixed** (F1); dense-only, see #2 |
| `write` | `writeLiveDocs` + `writeBits` | identical bytes; see #3 |
| — | `readSparseFixedBitSet`, `files` | **no Rust counterpart** — see #2; `files` is `SegmentCommitInfo`-level plumbing that lives in `lucene-index` |

### Findings

1. **[CORRECTNESS] A `.liv` with ghost bits set panicked instead of reporting
   corruption.** `writeBits` clears the bits of the last word past `maxDoc`
   before writing, and `new FixedBitSet(long[], int)` asserts they are clear on
   the way back in. Us: `FixedBitSet::from_words` carries the same
   `debug_assert`, so a corrupt `.liv` **panicked** in a debug build — and in a
   release build the inflated `cardinality()` made `max_doc - cardinality()`
   underflow. Either way a decoder was not reporting corruption through
   `Result`, which this port's decoders are required to do. **Fixed** —
   `parse` checks the last word explicitly and returns
   `Error::GhostBitsSet { max_doc }`. Tests:
   `ghost_bits_past_max_doc_are_rejected`,
   `a_full_last_word_is_fine_when_max_doc_is_a_multiple_of_64` (the no-ghost-bit-
   positions case must still accept a full last word).

2. **[INTENTIONAL] `readSparseFixedBitSet` / the 1 % `SPARSE_DENSE_THRESHOLD`
   is not ported.** Java picks `SparseFixedBitSet` vs `FixedBitSet` purely as
   an *in-memory* representation, after reading the identical dense bytes off
   disk. Both decode to the same bits. Already documented in the module header;
   confirmed correct this sweep.

3. **[INTENTIONAL] `writeBits`'s 1024-bit `Bits#applyMask` batching is not
   reproduced.** It exists to avoid per-bit calls through Java's generic `Bits`
   interface; this port's input is always a word-addressable `FixedBitSet`.
   Verified byte-identical: 1024 is a multiple of 64, so
   `sum(bits2words(chunk))` equals `bits2words(maxDoc)` for every `maxDoc`,
   which is exactly what `write` emits.

### Verdict

Swept clean; #1 fixed with tests, #2/#3 confirmed intentional and already
documented.

---

## `crates/lucene-codecs/src/doc_values_updates.rs`

Java counterparts: `index/DocValuesFieldUpdates.java`,
`index/NumericDocValuesFieldUpdates.java`,
`index/BinaryDocValuesFieldUpdates.java`.

**No Java byte-format counterpart exists**, and the module says so: the file
layout is this port's own invention (real Lucene's generation files are
`Lucene90DocValuesConsumer` output plus `SegmentCommitInfo.docValuesGen`
wiring, which this port does not have). What *is* portable is the **semantics**,
and that is what this section compares.

### Method correspondence

| Rust | Java | Verdict |
|---|---|---|
| `write_numeric_updates` | `DocValuesFieldUpdates.add(int, long)` + `finish()`'s stable sort | identical semantics (ascending by doc, last write per doc wins); reset support added (F1) |
| `read_numeric_updates` | `NumericDocValuesFieldUpdates.iterator()` | identical semantics; reset support added (F1) |
| `numeric_value_with_updates` | `ReadersAndUpdates`'s single-generation overlay read | divergent → **fixed** (F1) |
| `numeric_value_with_generations` | `DocValuesFieldUpdates.mergedIterator` (largest `delGen` wins per doc) | identical resolution order; divergent on reset → **fixed** (F1) |
| — | `BinaryDocValuesFieldUpdates` | **MISSING**, see #2 |
| — | `Iterator.asNumericDocValues`/`asBinaryDocValues`, `Container`, `Accountable`/`ramBytesUsed`, `PagedMutable` packed buffering | **no Rust counterpart** — Java-object-shape plumbing and heap accounting, not semantics |

### Findings

1. **[MISSING] `DocValuesFieldUpdates.reset(docId)` — an update that *removes*
   a doc's value — could not be expressed.** Java packs a `hasValue` bit
   alongside each buffered doc id (`HAS_VALUE_MASK`), exposes it as
   `Iterator.hasValue()`, and `ReadersAndUpdates` honours it. It is reachable
   from the public API: `IndexWriter.updateDocValues` with a null-valued field
   reaches `reset` (`IndexWriter.java:1770`,
   `FrozenBufferedUpdates.java:309`). Us: every entry was a bare `i64`, so a
   reset was indistinguishable from "this generation didn't touch the doc" and
   a removal silently fell through to the base value.
   **Fixed** — entries are `Option<i64>` end to end (`None` == reset), encoded
   as a per-entry `has_value` byte at a new `VERSION_HAS_VALUE`;
   `VERSION_START` files, which predate the byte, still read. A reset in the
   newest generation that touched a doc shadows both older generations and the
   base decode. Tests: `reset_entry_round_trips_as_a_none_value`,
   `reset_shadows_the_base_value`,
   `newest_generation_wins_whether_it_sets_or_resets`,
   `an_older_generations_reset_survives_a_disjoint_newer_generation`,
   `version_start_file_without_has_value_bytes_still_reads`.

   Two call sites outside this batch needed the type change and were updated
   minimally: `lucene_search::soft_deletes` (a reset now correctly means the
   doc's soft-deletes value was *removed*, so the doc is not soft-deleted and
   the base is not consulted — a semantic improvement, covered by the existing
   16 `soft_deletes` tests) and one `lucene_index::merge` test call.

2. **[MISSING, not fixed] `BinaryDocValuesFieldUpdates` has no counterpart.**
   Java supports updating BINARY doc values through the same
   `DocValuesFieldUpdates` machinery. **Recorded** — this is a whole second
   update type, not an edge case in the ported one, and the module's declared
   scope is explicitly the numeric chain. Sizing it belongs in a task, not a
   sweep fix.

3. **[INTENTIONAL] Resolution is a per-doc map lookup, not a merged
   priority-queue iterator.** Java's `mergedIterator` is a forward-only
   `PriorityQueue` merge over per-`delGen` iterators, because
   `ReadersAndUpdates` applies updates while streaming a segment rewrite. This
   port resolves one doc at a time against already-decoded `HashMap`s, which is
   what its random-access read API needs. Same newest-generation-wins outcome;
   `O(generations)` per lookup instead of amortized `O(log generations)` per
   doc across a full sweep — irrelevant at the generation counts this can
   reach.

### Verdict

Swept; #1 fixed with tests. Open: #2 (`BinaryDocValuesFieldUpdates`, a separate
feature), #3 intentional.

---

## Gates

- `cargo fmt --all` — clean.
- `cargo clippy -p lucene-codecs --all-targets -- -D warnings` — clean.
- `cargo test -p lucene-codecs` — all pass.
- `scripts/verify-write-path.sh` — 14/14 against real Lucene 10.5.0, including
  the new norms widths/sparse/multi-field segments and the 1500-term SORTED
  dictionary with full `lookupTerm` coverage.
- `docs/parity.md` updated in the same change (6 rows).

Two unrelated failures were observed in the workspace during this batch and are
**not** from these files: `lucene-index/tests/directory_fixtures.rs` (segments
file-name parsing, owned by an in-flight edit to
`lucene-store/src/directory.rs`) and a transient mid-refactor break in
`lucene-codecs/src/vectors.rs`. Both belong to concurrent sweep batches and
were left untouched.
