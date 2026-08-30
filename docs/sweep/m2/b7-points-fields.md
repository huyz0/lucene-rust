# b7-points-fields

Sweep of `crates/lucene-codecs/src/{points,field_infos,term_vectors,vectors}.rs`
against Lucene 10.5.0 at `/home/tuong/work/lucene`.

Java counterparts confirmed present in **`lucene/core`** (not backward-codecs)
for every format in this batch: `codecs/lucene90/Lucene90Points{Format,Reader,Writer}`,
`util/bkd/*`, `codecs/lucene94/Lucene94FieldInfosFormat`,
`codecs/lucene90/compressing/Lucene90CompressingTermVectors{Reader,Writer}`,
`codecs/lucene99/Lucene99{Flat,Hnsw}Vectors*`, `codecs/hnsw/*`, `util/hnsw/*`.
`lucene104` exists and adds only *scalar-quantized* vector formats
(`Lucene104ScalarQuantizedVectorsFormat`, `Lucene104HnswScalarQuantizedVectorsFormat`)
layered on the same lucene99 flat/HNSW pair — nothing in this batch is
superseded by it.

Totals: **28 findings** — 3 CORRECTNESS (all fixed), 7 MISSING (6 fixed;
the one recorded is HNSW, finding 24), 10 PERF (2 fixed with measurements,
8 reasoned/recorded), 8 INTENTIONAL.

---

## crates/lucene-codecs/src/points.rs

Java counterparts:
- `lucene/core/src/java/org/apache/lucene/codecs/lucene90/Lucene90PointsFormat.java`
- `lucene/core/src/java/org/apache/lucene/codecs/lucene90/Lucene90PointsReader.java`
- `lucene/core/src/java/org/apache/lucene/codecs/lucene90/Lucene90PointsWriter.java`
- `lucene/core/src/java/org/apache/lucene/util/bkd/{BKDConfig,BKDReader,BKDWriter,DocIdsWriter}.java`
- `lucene/core/src/java/org/apache/lucene/index/PointValues.java` (`intersect`, `Relation`, `IntersectVisitor`, `PointTree`)

### Method correspondence

| Rust | Java | Verdict |
|---|---|---|
| `open` | `Lucene90PointsReader(SegmentReadState)` | identical (codec names, VERSION_START=0/VERSION_CURRENT=1, `fieldNumber` as raw `readInt` terminated by `-1`, `< 0` ⇒ corrupt, trailing `indexLength`/`dataLength`, footer last) |
| `read_field_meta` | `BKDReader(metaIn, indexIn, dataIn)` | **was divergent** (no `BKDConfig` validation, no `minPackedValue <= maxPackedValue`) — fixed, finding 2/3 |
| `check_config` (new) | `BKDConfig`'s canonical constructor | identical bounds |
| `PointsReader::field` | `Lucene90PointsReader.getValues` | divergent by design: keyed by field *number*, no `FieldInfos` lookup / `IllegalArgumentException` on unindexed field (that check lives in `lucene-index`) |
| `decode_all_points` / `decode_leaves` | no Java counterpart (`PointValues` has no "give me everything" API) | not-in-Java |
| `intersect` (new) | `PointValues.intersect` + `BKDReader.BKDPointTree.{moveToChild,moveToSibling,moveToParent,visitDocIDs,visitDocValues}` | **was missing** — ported, finding 1 |
| `add_all` (new) | `BKDPointTree.addAll` | identical (doc ids only, no `compare`, no values) |
| `range_query` (new) | `PointRangeQuery`'s anonymous `IntersectVisitor` | identical relation logic |
| `decode_leaf_pointers` / `walk_node` | `BKDPointTree` traversal, collect-every-leaf specialisation | not-in-Java (this port's own eager shape) |
| `read_leaf_block` | `visitDocValuesWithCardinality` + `readCommonPrefixes` + `readCompressedDim` + `readMinMax` + `visit{Unique,Sparse,Compressed}RawDocValues` | identical, incl. field order (marker **before** the box), `-1`/`-2`/`>=0` dispatch, `Byte.toUnsignedInt(runLen)`, and both sub-block-count corruption errors |
| `read_doc_ids` | `DocIdsWriter.readInts(IndexInput,int,int[])` | identical (all 7 markers) |
| `read_legacy_delta_vint` | `readLegacyDeltaVInts` | identical |
| `read_bitset_ids` | `readBitSet` + `readBitSetIterator` | identical result; Java's `assert pos == count` is a hard error here (stricter, deliberate) |
| `read_delta_bpv16` | `readDelta16` + `decode16` | identical |
| `read_bpv21` | `readInts21` + `decode21` | identical (`l >> 42` is safe: bit 63 is provably 0 since ids are ≤ 21 bits) |
| `read_bpv24` | `readInts24` + `decode24` | identical |
| `floor_to_multiple_of_16` | `floorToMultipleOf16` | identical |
| `write` | `Lucene90PointsWriter.{writeField,finish}` | identical framing; `indexLength`/`dataLength` captured after both footers, as Java does |
| `write_field` | `BKDWriter.{finish,writeIndex}` | identical meta layout; **was missing** config validation — fixed, finding 4 |
| `get_num_left_leaf_nodes` | `BKDWriter.getNumLeftLeafNodes` | identical |
| `unsigned_byte_sub` | `NumericUtils.subtract` | identical for the comparison use |
| `widest_dim` | `BKDWriter.split` | divergent: only the second half (widest span); no `parentSplits` fairness pass — finding 9 |
| `compute_leaf_plan` | `BKDWriter.build` | same `numLeftLeafNodes`/`mid` arithmetic; different sort strategy — finding 10 |
| `pack_index` | `BKDWriter.{packIndex,recursePackIndex}` | identical, verified byte-level by the new fixture test (finding 1) |
| `write_leaf` | `BKDWriter.writeLeafBlock{Docs,PackedValues,...}` | divergent-but-legal: always prefix 0, always `compressedDim=-2`, runs of 1 — finding 11 |
| `write_leaf_doc_ids` | `DocIdsWriter.writeDocIds` | subset: only `CONTINUOUS_IDS`/`BPV_32` — finding 12 |

Java with **no** Rust counterpart: `BKDRadixSelector`, `HeapPointWriter`/
`HeapPointReader`, `OfflinePointWriter`/`OfflinePointReader`, `PointValue`,
`PointWriter`/`PointReader`, `MutablePointTreeReaderUtils`, `BKDUtil`
(SIMD compare helpers), `BKDWriter.merge`, `BKDWriter.writeField(MutablePointTree)`
(the one-dim `oneDimWriter` fast path), `DocIdsWriter.write*` for the four
encodings we never emit, `writeScalarInts24`/`readScalarInts24` and
`visitDocValuesNoCardinality` (pre-v7 layouts), `BKDReader.isTreeBalanced`/
`balanceTreeNodePosition`/`sizeFromBalancedTree` (pre-8.6 balanced trees),
`BKDPointTree.{size,estimatePointCount,estimateDocCount}`. All of these are
either legacy-format-only (this port reads BKD version 10 exclusively and
rejects everything else at the header, so they are unreachable) or belong to
the offline/mutable *sorting* machinery a large-scale writer needs and this
port's in-memory writer does not.

### Findings

1. **[MISSING → fixed]** *No pruning traversal at all.* Java's
   `PointValues.intersect` walks the packed index with the query's bounding
   box, skipping whole subtrees (`CELL_OUTSIDE_QUERY`), taking a doc-ids-only
   shortcut for wholly-matching ones (`CELL_INSIDE_QUERY` → `visitDocIDs`),
   and only decoding packed values where the cell straddles the boundary.
   This port had no equivalent: `decode_leaf_pointers` visited every node,
   read every node's `leftNumBytes` skip hint **and threw it away**, and
   decoded every leaf; `lucene-search`'s points query then filtered in
   memory. Consequence: every point query costs a full-field scan, and the
   entire `.kdi` — the reason the format exists — is dead weight.
   *Resolution*: ported `PointValues.intersect` as
   `PointsReader::intersect` + `Relation` + `IntersectVisitor`, plus a
   `range_query` convenience implementing `PointRangeQuery`'s visitor.
   Cell bounds are maintained exactly like `pushBoundsLeft`/`pushBoundsRight`/
   `popBounds`, and the split value is reconstructed from the packed index's
   prefix / first-diff-byte / `negativeDeltas` coding exactly like
   `BKDReader.readNodeData` — i.e. this is the read-side inverse of
   `pack_index`, and the first code in the module that *uses* split values
   rather than skipping them.
   Tests: `points.rs::intersect_tests` (1-dim boundary sweep vs brute force,
   2-dim, single-leaf tree, a duplicate-heavy tree that forces the packed
   index's `suffix == 0` "split value == last split value in this dim" branch,
   unknown field, plus assertions that pruning actually happened: a
   whole-range query compares exactly one cell and decodes zero packed values)
   and — the important one —
   `tests/points_fixtures.rs::intersect_over_real_lucene_packed_index_matches_brute_force`,
   which runs the traversal over **real `BKDWriter`-produced `.kdi` bytes**
   for all three fixture fields (1-dim `LongPoint`, 2-dim `IntPoint`,
   4-dim/2-index-dim shape) and compares against the manifest's Java-recorded
   values. A wrong split-value reconstruction prunes a subtree that contains
   matches and fails this test; nothing previously in the suite could catch
   that, because `decode_all_points` reads split descriptors only for their
   byte length.
   *Measured* (`cargo bench -p lucene-codecs --bench hot_paths -- points/`):
   on the Java fixture (1333 points, 3 leaves — pruning is capped at ~3x by
   construction) `range_query` 17.7 µs vs decode-all-and-filter 50.8 µs
   (2.9x, essentially the ceiling). On a synthetic 200k-point / 391-leaf tree
   at 0.1% selectivity: **15.55 µs vs 8.98 ms — 577x**.
   *Left open*: `lucene-search/src/points_query.rs` and
   `lucene-index/src/points_delete.rs` still call `decode_all_points`.
   Migrating them is b14/b11 work (different files); the API they need now
   exists and is fixture-verified.

2. **[MISSING → fixed]** *`.kdm` per-field config was never validated.*
   Java funnels `numDims`/`numIndexDims`/`bytesPerDim`/`maxPointsInLeafNode`
   through `BKDConfig.of(...)`, whose constructor throws on
   `numDims ∉ 1..16`, `numIndexDims ∉ 1..8`, `numIndexDims > numDims`,
   `bytesPerDim <= 0`, `maxPointsInLeafNode <= 0`; and asserts `numLeaves > 0`.
   We read all five as vints and used them directly. A corrupt or hostile
   `.kdm` with a negative `numIndexDims` reaches
   `vec![0u8; (num_index_dims * bytes_per_dim) as usize]` — a negative `i32`
   cast to `usize` is ~2^64 — and aborts the process on allocation failure
   instead of returning a decode error. `numLeaves` had the same problem via
   `Vec::with_capacity`. Note the `.kdm` footer is checked *last* (as in
   Java), so the checksum does not protect this.
   *Resolution*: `check_config` + a `numLeaves > 0` check in
   `read_field_meta`. Tests: `points.rs::config_validation_tests` (one per
   bound, each corrupting exactly one byte of an otherwise valid index).

3. **[MISSING → fixed]** *No `minPackedValue <= maxPackedValue` check.*
   `BKDReader`'s constructor throws `CorruptIndexException("minPackedValue …
   is > maxPackedValue … for dim=N")`, compared **per dimension** unsigned
   byte-wise. We accepted anything. With the new `intersect` this is no
   longer cosmetic: the root cell would be inverted and every query would
   silently return nothing.
   *Resolution*: `Error::MinGreaterThanMax`; test
   `min_packed_value_greater_than_max_rejected`.

4. **[MISSING → fixed]** *Write side had no `BKDConfig` validation.*
   `write(..., max_points_in_leaf_node = 0, ...)` reached
   `count.div_ceil(0)` and **panicked** (divide by zero) rather than
   returning an error; `num_dims > 16` / `num_index_dims > 8` /
   `bytes_per_dim <= 0` produced files real Lucene refuses to open.
   *Resolution*: same `check_config` on the write path. Tests:
   `write_rejects_{zero_max_points_in_leaf_node,zero_bytes_per_dim,too_many_dims,too_many_index_dims}`.

5. **[INTENTIONAL]** Only BKD version 10 (`VERSION_VECTORIZE_BPV24_AND_INTRODUCE_BPV21`)
   is accepted; `check_header(min=max=10)` rejects everything else. Java
   supports 4..10 with per-version branches (`isTreeBalanced`,
   `visitDocValuesNoCardinality`, `readScalarInts24`, implicit
   `numIndexDims`, `.kdi`-embedded `minLeafBlockFP`). Since this port only
   ever reads segments it or a current Lucene wrote, replicating five legacy
   layouts is scope, not fidelity. Consistent with the rest of the port.

6. **[INTENTIONAL]** `read_bitset_ids` materialises a `Vec<i32>` where Java
   returns a `DocBaseBitSetIterator` the visitor consumes lazily. Same doc
   ids; a bitset-shaped `DocIdSetIterator` has no consumer in this port yet.

7. **[PERF, recorded]** Java's `DocIdsWriter` owns a reusable
   `int[maxPointsInLeaf]` scratch and an `IntsRef`, and pushes whole runs
   into `visitor.visit(IntsRef)` to cut virtual-call overhead. Every Rust
   decoder here allocates a fresh `Vec<i32>` (and `read_bpv21`/`read_bpv24`
   a second scratch `Vec`) per leaf. At 512 ids/leaf that is 2 KB + 1.5 KB
   per leaf. Not fixed: it needs a reusable per-reader scratch buffer, which
   means `&mut self` on `intersect` (currently `&self`, so the reader is
   shareable). Bounded cost — measured `points/decode_all_points` is 34.7 µs
   for 3 leaves, i.e. allocation is not the dominant term at fixture scale —
   and the 577x win from finding 1 dwarfs it. Revisit if a profile ever shows
   it.

8. **[PERF, recorded]** `read_leaf_block` clones `scratch_value` into a fresh
   `Vec<u8>` per point (`Point { packed_value: Vec<u8> }`). Java hands the
   visitor a borrowed `byte[]` it must not retain. For a 512-point leaf of
   8-byte values that is 512 allocations vs 0. `intersect`'s
   `visit_with_value` path inherits this because it reuses `read_leaf_block`.
   Fixing it properly means a callback-shaped leaf decoder
   (`for_each_point(|doc, value: &[u8]|)`) that `decode_leaves` can still
   build `Vec`s on top of — contained, but it touches `check_index` and
   `merge`'s expectations, so it is left as a follow-up rather than done
   blind in this batch.

9. **[PERF/INTENTIONAL]** `widest_dim` implements only the second half of
   `BKDWriter.split`. Java first looks for a dimension split fewer than half
   as often as the most-split one (and not all-equal) and prefers it, to keep
   every dimension indexed. Ours always takes the widest span. Output is a
   valid, decodable tree either way (the split dimension is recorded per
   node), but for correlated dimensions our trees can be less selective on
   the neglected dimension, so `intersect` prunes less well there. Recorded,
   not fixed: `parentSplits` needs threading through `compute_leaf_plan`, and
   the win only shows on multi-dimension data this port does not yet
   generate at scale.

10. **[PERF]** `compute_leaf_plan` does a full `sort_by` at every split node
    (`O(n log² n)` overall) where `BKDWriter` uses `BKDRadixSelector`'s
    `O(n)` partial select (and spills to disk beyond `maxMBSortInHeap`).
    Also clones the whole point list up front and `split_off`s owned `Vec`s
    at each level. Already documented in the module; restated here with the
    complexity. Acceptable at current write volumes, and the shape (build a
    plan, then pack) is right; swapping `sort_by` for `select_nth_unstable_by`
    at each node would recover the `O(n log n)` bound without changing the
    tree, and is the natural follow-up.

11. **[INTENTIONAL]** `write_leaf` always writes common-prefix length 0 and
    `compressedDim = -2` with every run length 1, rather than computing the
    real common prefix / low-cardinality / high-cardinality choice. Legal
    (real Lucene's reader decodes it — proven by `VerifyPoints.java`), just
    larger on disk: for the fixture's 8-byte values this costs roughly the
    full `bytesPerDim` per point instead of the suffix. Compression ratio
    only, no correctness or decode-speed impact.

12. **[INTENTIONAL]** `write_leaf_doc_ids` emits only `CONTINUOUS_IDS` or
    `BPV_32`, never `BITSET_IDS`/`DELTA_BPV_16`/`BPV_21`/`BPV_24`. Always
    correct, up to 4x larger than Java's choice on the doc-id block. The read
    side decodes all seven markers, so nothing is lost on ingest of real
    Lucene files.

### Verdict

Swept clean on correctness and robustness. `intersect` closes the single
largest functional gap in the file. Open, all recorded above and none
blocking: callers still on `decode_all_points` (b11/b14), per-leaf scratch
reuse (7), per-point `Vec` clone (8), `parentSplits` (9), radix select (10).

---

## crates/lucene-codecs/src/field_infos.rs

Java counterparts:
- `lucene/core/src/java/org/apache/lucene/codecs/lucene94/Lucene94FieldInfosFormat.java`
- `lucene/core/src/java/org/apache/lucene/index/{FieldInfo,FieldInfos,IndexOptions,DocValuesType,DocValuesSkipIndexType,VectorEncoding,VectorSimilarityFunction}.java`

### Method correspondence

| Rust | Java | Verdict |
|---|---|---|
| `parse` | `Lucene94FieldInfosFormat.read` | identical field-for-field; **was missing** the `new FieldInfos(infos)` cross-field checks — fixed, finding 13 |
| `write` | `Lucene94FieldInfosFormat.write` | identical inverse (always emits `FORMAT_DOCVALUE_SKIPPER` = 2) |
| `IndexOptions::{from_byte,to_byte}` | `getIndexOptions`/`indexOptionsByte` | identical, all 6 values incl. `DOCS_AND_CUSTOM_FREQS` = 5 |
| `IndexOptions::subsumes_positions/offsets` | `IndexOptions.subsumes` | identical, incl. the `DOCS_AND_CUSTOM_FREQS` special case (does *not* subsume positions) |
| `DocValuesType::{from_byte,to_byte}` | `getDocValuesType`/`docValuesByte` | identical, all 6 |
| `DocValuesSkipIndexType::{from_byte,to_byte,is_compatible_with}` | `getDocValuesSkipIndexType` / `isCompatibleWith` | identical |
| `VectorEncoding::{from_byte,to_byte}` | `getVectorEncoding` | identical (BYTE=0, FLOAT32=1) |
| `VectorSimilarityFunction::{from_byte,to_byte}` | `getDistFunc` | identical (EUCLIDEAN/DOT_PRODUCT/COSINE/MAXIMUM_INNER_PRODUCT = 0..3) |
| `FieldInfo::check_consistency` | `FieldInfo.checkConsistency` | identical, check for check, in Java's order |
| `FieldInfos::check_consistency` (new) | `FieldInfos(FieldInfo[])` constructor | ported, finding 13 |
| `FieldInfos::field_by_number` | `FieldInfos.fieldInfo(int)` | same answer, `O(n)` vs `O(1)` — finding 16 |
| `FieldInfos::field_by_name` (new) | `FieldInfos.fieldInfo(String)` | added |
| `FieldInfos::soft_deletes_field` / `parent_field` (new) | `getSoftDeletesField` / `getParentField` | added |

Every bit is accounted for: `0x1` STORE_TERMVECTOR, `0x2` OMIT_NORMS,
`0x4` STORE_PAYLOADS, `0x8` SOFT_DELETES_FIELD, `0x10` PARENT_FIELD_FIELD
(format ≥ 1), `0x20` DOCVALUES_SKIPPER (format ≥ 2), `0xC0` always-unused,
and both "bit set but format too old" rejections — matching Java exactly,
including that `parent_field` is forced false (not rejected) when
`format < FORMAT_PARENT_FIELD` while `0xF0` is separately rejected.

Java with **no** Rust counterpart: `FieldInfos`' aggregate flags
(`hasPostings`/`hasProx`/`hasFreq`/`hasOffsets`/`hasNorms`/`hasDocValues`/
`hasPointValues`/`hasVectorValues`/`hasTermVectors`/`hasPayloads`), its
`byNumber` array / `byName` map, `FieldInfos.getMergedFieldInfos`,
`FieldInfos.Builder`/`FieldNumbers` (the writer-side number allocator), and
`FieldInfo.verifySameSchema` (a merge-time cross-segment check). The
aggregates are a scan away via the accessors above and nothing consumes them
yet; the builder/merge machinery belongs to `lucene-index` (b9/b10).

### Findings

13. **[MISSING → fixed]** *The `FieldInfos` constructor's cross-field checks
    were not ported.* `Lucene94FieldInfosFormat.read` ends in
    `return new FieldInfos(infos)`, and that constructor throws
    `IllegalArgumentException` on duplicate field names, duplicate field
    numbers, more than one soft-deletes field, and more than one parent
    field. So a `.fnm` tripping any of these is **rejected by real Lucene at
    read time**; we accepted it silently, and `field_by_number`'s
    first-match-wins scan would then quietly shadow one of two same-numbered
    fields — mis-decoding every other per-field file in the segment.
    *Resolution*: `FieldInfos::check_consistency`, called at the end of
    `parse`. Tests: `duplicate_field_names_rejected`,
    `duplicate_field_numbers_rejected`, `multiple_soft_deletes_fields_rejected`,
    `multiple_parent_fields_rejected`, plus
    `distinct_soft_deletes_and_parent_fields_accepted` for the legal
    neighbouring case (one field of each is fine; the *same* field being both
    is caught by `FieldInfo::check_consistency`, which we already had).

14. **[INTENTIONAL]** `write` does not re-run `check_consistency`. Java's
    `Lucene94FieldInfosFormat.write` doesn't either — the invariants are
    enforced by `FieldInfo`'s own setters/constructor upstream. Round-trip
    tests go through `parse`, which does validate.

15. **[PERF]** Java shares the previous field's attribute map when equal
    (`if (attributes.equals(lastAttributes)) attributes = lastAttributes;`)
    to avoid one `HashMap` per field. We allocate a `Vec<(String, String)>`
    per field regardless. `.fnm` has one entry per *field*, not per document,
    so this is tens of allocations per segment open — genuinely negligible,
    and the sharing trick only pays off in Java because a `HashMap` is far
    heavier than a small `Vec`. Recorded, not fixed.

16. **[PERF]** `field_by_number` / `field_by_name` are linear scans; Java
    uses an array indexed by field number and a `HashMap` by name. With ~87
    call sites across the workspace and typical field counts in the tens,
    the scan is likely faster than a hash lookup for `by_name` and comparable
    for `by_number`; the crossover is somewhere around 30-50 fields. Not
    fixed: adding a `Vec<Option<u32>>` index would be right for a
    hundreds-of-fields schema but is speculative today, and the accessor
    signatures do not change if it is added later.

### Verdict

Swept clean. Format decoding was already exact; the gap was the `FieldInfos`-
level validation Java performs during read, now closed. Open: nothing
blocking (15, 16 recorded).

---

## crates/lucene-codecs/src/term_vectors.rs

Java counterparts:
- `lucene/core/src/java/org/apache/lucene/codecs/lucene90/Lucene90TermVectorsFormat.java`
- `lucene/core/src/java/org/apache/lucene/codecs/lucene90/compressing/Lucene90CompressingTermVectorsReader.java`
- `lucene/core/src/java/org/apache/lucene/codecs/lucene90/compressing/Lucene90CompressingTermVectorsWriter.java`
- `lucene/core/src/java/org/apache/lucene/index/{Fields,Terms,TermsEnum}.java`

### Method correspondence

| Rust | Java | Verdict |
|---|---|---|
| `open` | `Lucene90CompressingTermVectorsReader(...)` ctor | identical field order and the first two dirty-chunk checks; **was missing** the third — fixed, finding 18 |
| `TermVectorsReader::document` | `get(int doc)` (returns `TVFields`) | same decoded content; decodes the whole chunk rather than skipping to one doc — finding 20 |
| `document`'s distinct-field-number block | `flushFieldNums`'s inverse (token `(min(n-1,7)<<5) \| bitsRequired`, overflow vint, headerless `PackedInts.PACKED`) | identical |
| `document`'s flags block | `readVInt()` selector 0 (dedup per distinct field number) / 1 (direct per field) | identical, both branches, incl. rejecting any other selector |
| `read_length_prefixed_slice` | `slice(IndexInput)` | identical (`writeVLong`/`readVInt` agree below 2^31) |
| `direct_writer_bits_required` | `DirectWriter.bitsRequired` + `roundBits` | identical |
| `build_field` | `TVFields`/`TVTerms`/`TVPostingsEnum` materialisation | identical: prefix/suffix term reconstruction, per-term position delta chains, `charsPerTerm * rawPositionDelta` offset patching applied **before** positions are delta-decoded (matching Java's ordering exactly), `length += prefixLen + suffixLen`, payload slicing |
| `write_best_speed` | `Lucene90CompressingTermVectorsWriter.flush` | scoped down (finding 21); **had a real encoding bug** — fixed, finding 17 |
| `encode_literal_lz4` | `CompressionMode.FAST` compressor | legal single-literal-run LZ4; no matching |

Java with **no** Rust counterpart: `TermsEnum`/`PostingsEnum` iteration
(`TVTerms.iterator`, `TVTermsEnum.{seekCeil,next,postings}`) — this port
returns a fully materialised `TermVectorsDocument` instead, which
`lucene-search/src/term_vectors_query.rs` consumes directly;
`Lucene90CompressingTermVectorsWriter.merge`'s bulk chunk copy;
`checkIntegrity`; the `prefetch`/`BlockState` block cache; multi-chunk
flushing and `numDirtyChunks` accounting on the write side.

### Findings

17. **[CORRECTNESS → fixed]** *Writer emitted offset/payload streams for a
    flagged-but-termless field.* `write_best_speed` gated the `charsPerTerm`
    array + the two offset streams on `all_fields.iter().any(|f| f.has_offsets)`
    and the payload-length stream on the analogous `has_payloads` flag. Both
    readers — Java's and ours — instead gate on `totalOffsets > 0` /
    `totalPayloads > 0`, which are **sums of term frequencies**. A field that
    carries the OFFSETS (or PAYLOADS) flag but holds no terms contributes
    zero to those sums, so the writer emitted blocks the reader never
    consumes and every following byte of the chunk decoded as garbage
    (in practice: a panic or a nonsense term). Reachable from `merge.rs`'s
    term-vector merge and from any hand-built `TermVectorsDocument`.
    *Resolution*: gate on `!start_offsets_flat.is_empty()` /
    `!payload_lengths_flat.is_empty()`, which are exactly `totalOffsets` and
    `totalPayloads`. Test:
    `write_best_speed_flagged_but_termless_field_round_trips` — verified to
    fail against the old gating and pass with the fix.

18. **[MISSING → fixed]** *Third dirty-chunk corruption check absent.*
    Java rejects `numDirtyDocs < numDirtyChunks` ("Cannot have more dirty
    chunks than documents within dirty chunks") — every forced flush
    contributes at least one doc. We had the other two checks but not this
    one. *Resolution*: `Error::MoreDirtyChunksThanDirtyDocs`. Tests:
    `more_dirty_chunks_than_dirty_docs_rejected` and the boundary case
    `equal_dirty_chunks_and_dirty_docs_accepted`. Reaching it required a
    fixture claiming ≥ 2 chunks, so the test builder gained an
    `index_num_chunks` parameter.

19. **[PERF → fixed]** `build_field` allocated `Vec::with_capacity(freq)` for
    positions, start offsets, end offsets and payloads *unconditionally*,
    then left three of the four empty for a typical positions-only field.
    `with_capacity` allocates; `Vec::new()` does not. Now each is allocated
    only when its flag is set — three fewer heap allocations per term on the
    common path.

20. **[PERF, recorded]** `document(doc)` decodes **every** document in the
    chunk (all field metadata arrays, all terms, all positions/offsets/
    payloads, the whole LZ4 unit) and then materialises only the requested
    doc's fields. Java's `get(doc)` does the same *reads* for the packed
    metadata arrays (they are not seekable) but skips the requested doc's
    neighbours when materialising, and — crucially — decompresses only the
    `[docOff+payloadOff, +docLen+payloadLen)` slice of the LZ4 unit rather
    than the whole thing. With Lucene's default 4 KB / 128-doc chunks the
    waste is bounded by the chunk, and this port writes single-chunk files
    anyway (finding 21), where it is exactly zero extra work for a
    one-document segment and grows linearly for larger ones. Already
    documented in the module as the port-wide decode-fully trade-off; not
    changed here because the fix (partial LZ4 decompression with a skip
    prefix) belongs with the `lz4` module (b3).

21. **[INTENTIONAL]** Writer scope: single chunk (`chunk_docs = docs.len()`),
    no term prefix sharing (`prefix_len` always 0), fixed
    `charsPerTerm = 1.0`, per-array exact bit widths rather than
    cross-chunk minimisation, one literal-run LZ4 block. Produces valid,
    Java-openable files; costs compression ratio only. Note the
    `charsPerTerm` constant is provably free: the read side multiplies by the
    same value it divided by.

22. **[INTENTIONAL]** Distinct field numbers are written in **first-seen**
    order; `flushFieldNums` sorts them. The reader (ours and Java's
    `TVFields.terms`/`iterator`) resolves fields through `fieldNumOffs`
    indices and uses a linear scan, not `binarySearch`, so both orders decode
    identically — `Arrays.binarySearch(fieldNums, ...)` appears only on
    Java's *writer* side. Bit width is unaffected (it is derived from the
    maximum either way). Left as is rather than churn a fixture that a Java
    verifier has already signed off on.
    *Caveat worth recording for whoever wires the flush path*: real Lucene's
    `CheckIndex` iterates term-vector fields via `TVFields.iterator()`,
    which yields them in **document field order**, and requires strictly
    ascending field *names*. Callers of `write_best_speed` must therefore
    supply each document's fields in name order. Nothing here can enforce
    that (this module never sees names).

### Verdict

Swept clean; one real writer bug found and fixed, one corruption check
restored. Open: partial-chunk decode (20), writer compression scope (21),
and the field-name-ordering caller contract noted in 22.

---

## crates/lucene-codecs/src/vectors.rs

Java counterparts (all present, all in `lucene/core`):
- `codecs/lucene99/Lucene99FlatVectorsFormat.java` (`.vec` + `.vemf`), `Lucene99FlatVectorsReader.java`, `Lucene99FlatVectorsWriter.java`
- `codecs/lucene99/Lucene99HnswVectorsFormat.java` (`.vex` + `.vem`), `Lucene99HnswVectorsReader.java`, `Lucene99HnswVectorsWriter.java`
- `codecs/lucene95/{OffHeapFloatVectorValues,OrdToDocDISIReaderConfiguration}.java`
- `codecs/hnsw/{FlatVectorsFormat,FlatVectorsReader,FlatVectorsWriter,FlatVectorsScorer,DefaultFlatVectorScorer}.java`
- `util/hnsw/{HnswGraph,HnswGraphBuilder,HnswGraphSearcher,NeighborQueue,NeighborArray,OnHeapHnswGraph,RandomVectorScorer,...}.java`
- `index/VectorSimilarityFunction.java`, `util/VectorUtil.java`
- `codecs/lucene104/*` — newer, quantized-only; layers on the same lucene99 pair

### Method correspondence

| Rust | Java | Verdict |
|---|---|---|
| `write_vectors` | `Lucene99FlatVectorsWriter.{addField,flush,finish}` | **not a port** — this port's own `.vec`/`.vem` format (finding 23) |
| `FlatVectorsReader::open` | `Lucene99FlatVectorsReader` ctor + `readFields` | not a port, same reason |
| `VectorField::vector(doc)` | `FloatVectorValues.vectorValue(ord)` + `OrdToDocDISIReaderConfiguration` | divergent: linear scan by doc id vs an ordinal-indexed off-heap slice behind a DISI — finding 26 |
| `VectorField::search` | `Lucene99HnswVectorsReader.search` | **divergent algorithm**: exhaustive scan vs HNSW graph search — finding 24. Now rejects a wrong-dimension query — finding 25 |
| `VectorSimilarityFunction::score` | `VectorSimilarityFunction.compare(float[], float[])` | identical formulas for all four; `Cosine` now clamps like `normalizeToUnitInterval` |
| `square_distance` / `dot_product` / `cosine` | `VectorUtil.{squareDistance,dotProduct,cosine}` | identical results up to summation order; now lane-split — finding 27. `cosine`'s divisor now matches Java's single `f64` sqrt of the product |
| `scale_max_inner_product_score` | `VectorUtil.scaleMaxInnerProductScore` | identical |
| `ScoredDoc` + `BinaryHeap` | `NeighborQueue` / `KnnCollector` | equivalent bounded min-heap |

Java with **no** Rust counterpart — the entire HNSW stack:
`HnswGraph`, `HnswGraphBuilder` (`DEFAULT_MAX_CONN = 16`,
`DEFAULT_BEAM_WIDTH = 100`), `HnswGraphSearcher`/`AbstractHnswGraphSearcher`/
`FilteredHnswGraphSearcher`/`SeededHnswGraphSearcher`, `OnHeapHnswGraph`,
`NeighborArray`, `NeighborQueue`, `FloatHeap`, `HnswGraphMerger`/
`IncrementalHnswGraphMerger`/`ConcurrentHnswMerger`, `HnswUtil`,
`RandomVectorScorer`(`Supplier`), `Lucene99HnswVectorsWriter`'s graph
serialisation and `Lucene99HnswVectorsReader`'s `.vex` offset/neighbour
decoding, `OrdToDocDISIReaderConfiguration` (the sparse ord→doc mapping),
`OffHeapFloatVectorValues`/`OffHeapByteVectorValues`/`OffHeapFloat16VectorValues`,
byte and float16 vector encodings, and every scalar-quantized format in
`lucene104`.

### Findings

23. **[INTENTIONAL, pre-existing and documented]** *This is not Lucene's
    vector format.* `.vec`/`.vem` here are this port's own layout
    (`LuceneRustFlatVectorsData`/`Meta` codec names), not
    `Lucene99FlatVectorsFormat`'s. Real `.vec` is raw vectors addressed by
    *ordinal* with a separate `.vemf` meta carrying an
    `OrdToDocDISIReaderConfiguration` (dense fast path, or an `IndexedDISI` +
    `DirectMonotonicReader` ord→doc mapping for sparse fields); this port
    writes an explicit per-vector doc-id list instead. Consequence: **a
    Lucene-written vector field cannot be read by this module, and a
    Rust-written one cannot be read by Lucene** — unlike every other format
    in this crate. Restated here because it is the dominant fact about the
    file and it is easy to miss behind the shared `codec_util` framing.
    Also: `FieldVectors` carries its own `dimension`/`similarity`, decoupled
    from `field_infos::FieldInfo::{vector_dimension,vector_similarity_function}`;
    nothing cross-validates them.

24. **[MISSING, recorded — not fixed]** *HNSW graph search is not ported at
    all; `search` is a brute-force scan.* Stated plainly, as asked:
    `VectorField::search` scores **every** stored vector against the query
    and keeps a bounded heap — `O(n·d)` per query, exact. Java's
    `Lucene99HnswVectorsReader.search` walks a multi-layer HNSW graph
    (`HnswGraphSearcher`), touching roughly `beamWidth + M·log n` candidates
    — approximate, with recall traded for speed. There is no `.vex` file,
    no graph construction, no layer assignment, no neighbour diversity
    pruning, no merge-time graph reuse.
    *What it costs*: measured, `vectors/brute_force_search_50k_dim128_k10`
    is **2.32 ms** per query for 50k 128-dim vectors on one core (~430
    queries/s). Cost is exactly linear in `n·d`: 1M 768-dim vectors would be
    ≈ 280 ms per query. An HNSW search over the same 50k set visits on the
    order of 1-3k candidates rather than 50k — roughly **20-30x fewer
    distance computations here, and ~300x at 1M vectors**, with the gap
    widening as `n` grows because brute force is `O(n)` and HNSW is
    `O(log n)`. Exact search also has a real advantage (100% recall) that
    matters for small fields; Lucene itself brute-forces below
    `HNSW_GRAPH_THRESHOLD`.
    *Not fixed*: porting `HnswGraphBuilder` + `HnswGraphSearcher` +
    `.vex`/`.vem` serialisation is a multi-day module of its own, larger than
    everything else in this batch combined, and it is gated on finding 23 —
    there is no point building a graph over a non-Lucene flat format. Carry
    forward as a dedicated task; the honest current status is "exact KNN over
    small fields, unusable at scale".

25. **[CORRECTNESS → fixed]** *Wrong-dimension queries scored silently.*
    `score` zipped `query` with each stored vector, so a query of the wrong
    length was scored over the shorter of the two and returned a
    plausible-looking number. Every one of Java's scoring primitives
    (`VectorUtil.dotProduct`/`cosine`/`squareDistance`) throws
    `IllegalArgumentException("vector dimensions differ: A!=B")` instead.
    *Resolution*: `Error::QueryDimensionMismatch`, checked at the top of
    `search` (which is now `Result`-returning). Test:
    `search_rejects_query_of_the_wrong_dimension`.

26. **[CORRECTNESS → fixed]** *`cosine` accumulated differently from Java.*
    Java computes `(float)(sum / Math.sqrt((double) norm1 * (double) norm2))`
    — one square root, in double precision, of the *product*. We took two
    separate `f32` square roots and multiplied them: three roundings instead
    of one, drifting a few ulps from Java. Invisible in a round-trip test,
    visible the moment two near-identical candidates are ranked against each
    other. Also added `VectorUtil.normalizeToUnitInterval`'s `max(_, 0)`
    clamp to the `Cosine` branch, which we had on `DotProduct` but not
    `Cosine` (Java has it on both). The zero-vector → `0.0` guard is kept and
    now explicitly documented as this port's own divergence: Java returns
    `NaN` there and relies on index-time validation we do not have yet.

27. **[PERF → fixed, measured]** *Distance kernels were serial scalar loops.*
    `a.iter().zip(b).map(..).sum()` builds a single-accumulator dependency
    chain that LLVM may not reassociate (float addition is not associative),
    so the loop stayed scalar — while Java runs these on the Panama Vector
    API (`PanamaVectorUtilSupport`, 8-wide) and even its scalar fallback
    unrolls 2x. Rewrote `square_distance`, `dot_product` and `cosine` as
    eight independent accumulators over `chunks_exact(8)` (which also drops
    the bounds checks; an indexed 8-lane version measured *slower* than the
    original, at 2.60 ms, before switching to `chunks_exact`).
    *Measured* on `vectors/brute_force_search_50k_dim128_k10`:
    **2.32 ms → 1.62-1.76 ms (~1.3-1.4x)**. Explicit `std::simd` would go
    further but needs nightly or a new dependency — an architecture decision,
    not a batch-local one. The lane split changes summation order and hence
    the last ulp, exactly as switching between Lucene's own two
    implementations does.

28. **[PERF, recorded]** `VectorField::vector(doc_id)` is a linear scan over
    `entries`; Java resolves doc→ord through `OrdToDocDISIReaderConfiguration`
    (an `IndexedDISI` + `DirectMonotonicReader`, or a bare identity mapping
    for dense fields) and then indexes an off-heap slice — `O(1)`-ish and
    zero-copy. Ours is `O(n)` *and* holds every vector on the heap as a
    `Vec<Vec<f32>>` (one allocation per vector; Java holds one `IndexInput`).
    For a 1M x 768 field that is 1M allocations and ~3 GB resident. Not
    fixed — it is inseparable from finding 23 (the format itself has no
    ordinal addressing), and both should be redone together when the real
    `Lucene99FlatVectorsFormat` is ported.

### Verdict

**Not swept clean, by design.** Three fixes landed (dimension check, cosine
precision + clamp, SIMD-shaped kernels), but the file remains a
non-Lucene-compatible flat store with brute-force search. The two open items
are large and coupled: port the real `Lucene99FlatVectorsFormat` (23, 28),
then the HNSW graph on top of it (24). Neither is a defect in what is here;
both are unported scope, now measured rather than asserted.

---

## Carry-over for later batches

- [ ] `lucene-search/src/points_query.rs` and `lucene-index/src/points_delete.rs`
      still decode every point and filter in memory. `PointsReader::intersect`/
      `range_query` now exist and are fixture-verified; migrating the callers
      is a 577x win at 200k points. (Owners: b14, b11.)
- [ ] Points: reusable per-leaf doc-id scratch (finding 7) and a
      callback-shaped leaf decoder to stop cloning a `Vec<u8>` per point
      (finding 8).
- [ ] Points writer: `select_nth_unstable_by` instead of `sort_by` per split
      node (finding 10); `parentSplits` fairness in `widest_dim` (finding 9).
- [ ] Term vectors: partial LZ4 decompression so `document(doc)` stops
      inflating the whole chunk (finding 20) — needs `lz4` module support (b3).
- [ ] Term vectors: callers of `write_best_speed` must pass each document's
      fields in ascending field-*name* order or real `CheckIndex` rejects the
      segment (finding 22). Enforce it where names are known (b9's flush path).
- [ ] **Vectors, large**: port `Lucene99FlatVectorsFormat` (`.vec`/`.vemf`
      with `OrdToDocDISIReaderConfiguration`) so vector fields interoperate
      with real Lucene at all (findings 23, 28), then
      `Lucene99HnswVectorsFormat` + `util/hnsw/*` on top (finding 24). Until
      then vector search is exact but linear: 2.32 ms/query at 50k x 128,
      ~280 ms/query extrapolated to 1M x 768.
