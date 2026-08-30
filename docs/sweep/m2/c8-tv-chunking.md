# c8-tv-chunking — M2 sweep follow-up

Closes the two carry-overs named in the batch brief:

- **`c4-merge-fastpath` finding 13** — `Lucene90CompressingTermVectorsWriter.merge`'s
  bulk path could not be ported because
  `term_vectors::write_best_speed` wrote the **whole segment as one chunk**,
  so there was no chunk-appending writer to append to and no multi-chunk
  source to copy from.
- **`b5-postings` F6** — no `PostingsEnum`-flags plumbing, so frequencies were
  decoded even for a consumer that asked for `PostingsEnum.NONE`/`DOCS`.

Java source of truth: `/home/tuong/work/lucene` (Lucene 10.5.0) —
`codecs/lucene90/Lucene90TermVectorsFormat.java`,
`codecs/lucene90/compressing/Lucene90CompressingTermVectorsWriter.java`,
`codecs/lucene90/compressing/Lucene90CompressingTermVectorsReader.java`,
`codecs/compressing/MatchingReaders.java`,
`codecs/lucene104/Lucene104PostingsReader.java`,
`codecs/lucene104/PForUtil.java`, `codecs/lucene104/PostingsUtil.java`,
`index/PostingsEnum.java`,
`codecs/lucene90/compressing/Lucene90CompressingStoredFieldsWriter.java`
(as the already-ported shape `c4` established).

## Files swept / changed

| File | Owner | What changed |
|---|---|---|
| `crates/lucene-codecs/src/term_vectors.rs` | b7's, not under concurrent edit | rewritten write side: `TermVectorsWriter` (streaming, chunking, `copy_chunks`); reader gained `DecodedChunk`/`ChunkCursor`, `chunk_for_doc`, `check_integrity` and the four `.tvm` accessors it discarded |
| `crates/lucene-codecs/src/postings.rs` | b5's | `PostingsFlags`, `*_with_flags` entry points, `needs_freq` through both block decoders; one real corruption error replacing a `debug_assert` |
| `crates/lucene-codecs/src/for_util.rs` | b2's | **additive**: `pfor_skip` (`PForUtil.skip`) |
| `crates/lucene-codecs/src/blocktree.rs` | c1's (swept) | **additive**: `postings_with_flags`, `lazy_postings_with_flags` |
| `crates/lucene-codecs/src/postings_writer.rs` | b5's | tests only |
| `crates/lucene-index/src/merge.rs` | unowned; edits scoped to the term-vectors merge | `write_merged_term_vectors` (replaces `merge_term_vectors`), `TermVectorsMergeStrategy`, `term_vectors_merge_strategy` |
| `crates/lucene-index/src/term_delete.rs` | b11's | one call site now asks for `PostingsFlags::DocsOnly` |
| `crates/lucene-index/src/index_writer.rs` | c7's | three doc-comment references to the renamed function |
| `crates/lucene-index/examples/write_merged_segment_fixture.rs` | c4's | stores term vectors; also repaired against c7's `delete_documents` rename |
| `crates/lucene-codecs/examples/write_term_vectors_fixture.rs`, `fixtures/src/VerifyTermVectors.java` | — | new `_2` multi-chunk segment with offsets/payloads |
| `fixtures/src/VerifyMergedSegment.java` | c4's | per-document term-vector recomputation |
| `benchmarks/rust-runner/src/merge_bench.rs` | c4's | term-vector and postings-flags scenarios |
| `docs/parity.md` | — | both rows rewritten |

---

## Headline measurements

`benchmarks/rust-runner/src/merge_bench.rs`, extended with a `term vectors`
and a `postings flags` scenario. 4 segments x 20 000 documents; each
term-vector document has two fields, ~60 bytes of term text, positions on both
and offsets + payloads on one (865 kB of `.tvd` per segment). Best of three
after a warm-up, except the quadratic arm noted below.

**Every "before" figure is re-run, not remembered**, and the term-vector ones
are the *actual* pre-`c8` code rather than an approximation of it: finding 8's
`TermVectorsWriter::with_geometry` is `Lucene90CompressingTermVectorsWriter`'s
own `chunkSize`/`maxDocsPerChunk` constructor parameters, so setting them past
the segment reproduces exactly what `write_best_speed(chunk_docs = docs.len())`
did.

| Path | before | after | speedup |
|---|---|---|---|
| Term-vector **flush write** (20 000 docs) | 15.4 ms / 1.30 M docs/s | **12.0 ms / 1.67 M docs/s** | **1.28x** |
| `.tvd` size for the same documents | 864 869 B (1 chunk) | 869 211 B (160 chunks) | +0.5 % |
| **Random-access `document()`**, 200 scattered reads | 625.8 ms | **3.2 ms** | **195x** |
| Term-vector **merge, BULK** (4 x 20 000, matching + deletion-free) | 289 292 ms | **0.6 ms** | **469 076x** |
| Term-vector **merge, per-document** (a source with deletions or renumbered fields) | 289 292 ms | **113.5 ms** | **2 548x** |
| — of which the byte copy alone (BULK vs per-document, both post-`c8`) | 113.5 ms | **0.6 ms** | **189x** |
| Docs-only **lazy cursor** walk, freqs 1–3 | 105.6 µs | 98.6 µs | 1.07x |
| Docs-only **eager** `read_postings`, freqs 1–3 | 41.3 µs | 34.0 µs | 1.22x |
| Docs-only **lazy cursor** walk, freqs 1–48 | 47.4 µs | 42.2 µs | 1.12x |
| Docs-only **eager** `read_postings`, freqs 1–48 | 19.2 µs | 14.6 µs | 1.32x |

Reading these honestly:

- **The merge figure decomposes into two independent wins, and the headline
  ratio is dominated by the smaller one.** The pre-`c8` merge was quadratic,
  not merely slow: `reader.document(doc)` on a one-chunk segment decodes the
  *entire* segment, so merging 80 000 documents cost 80 000 whole-segment
  decodes — 289 seconds, and growing with the square of the segment. Chunking
  plus the `ChunkCursor` alone turns that into 113.5 ms (**2 548x**), and that
  is the part that matters for correctness of the comparison. The byte copy on
  top is a further **189x**, which is the number directly comparable to c4's
  stored-fields BULK-vs-DOC ratio (there: 9.7 ms vs 0.8 ms). The 469 076x
  headline is real but is mostly the quadratic going away.
  (The quadratic arm is measured with a single timed pass and no warm-up
  rep — a second pass would double a five-minute measurement to no purpose.)
- **Write throughput improved 1.28x while doing strictly more work** (prefix
  compression, the derived `charsPerTerm`, sorting the field numbers, 160
  chunk headers instead of 1). The one-chunk writer built a single
  segment-sized suffix buffer and ran one segment-sized LZ4 literal run; the
  chunked writer works in 4 kB units that stay in cache.
- **The file grew 0.5 %**, not the several percent 160 extra chunk headers and
  160 extra `.tvx` entries would cost on their own — findings 4, 5 and 6
  (prefix compression, the real `charsPerTerm`, the per-field-number flags
  array) nearly pay for the chunking.
- **The postings-flags win is modest and that is the honest result.** Doc
  deltas, not frequencies, dominate a block's decode, so `PForUtil.skip`ping
  the frequency block saves 7–32 % depending on how wide that block is — which
  is why the benchmark measures two corpora, one whose frequencies need ~2 bits
  and one ~6. It is the same shape of saving Lucene takes, at the same place;
  it is not a step change, and nothing in this batch claimed it would be.

c4's own figures were re-run unchanged in the same process (stored fields
586x / 31x / 26x, postings 11.9x, BKD 2.7x), so nothing in this batch regressed
them.

---

## `crates/lucene-codecs/src/term_vectors.rs`

Java counterparts:
- `lucene/core/src/java/org/apache/lucene/codecs/lucene90/Lucene90TermVectorsFormat.java`
- `lucene/core/src/java/org/apache/lucene/codecs/lucene90/compressing/Lucene90CompressingTermVectorsWriter.java`
- `lucene/core/src/java/org/apache/lucene/codecs/lucene90/compressing/Lucene90CompressingTermVectorsReader.java`

The format's own constants, read out of `Lucene90TermVectorsFormat`'s sole
constructor rather than assumed:
`super("Lucene90TermVectorsData", "", CompressionMode.FAST, 1 << 12, 128, 10)`
— `chunkSize = 4096`, `maxDocsPerChunk = 128`, `blockShift = 10`.

### Method correspondence (write side, all new)

| Rust | Java | Verdict |
|---|---|---|
| `TermVectorsWriter` (the object) | `Lucene90CompressingTermVectorsWriter` | **new** — was a one-shot free function `write_best_speed(&[TermVectorsDocument])` |
| `TermVectorsWriter::new` / `::with_geometry` | the format's fixed geometry / the writer's `chunkSize`/`maxDocsPerChunk`/`blockShift` constructor parameters | identical |
| `add_document` | `startDocument` + `startField`/`startTerm`/`addPosition` + `finishDocument` | equivalent: one call for a whole document, since this port's input is a materialised `TermVectorsDocument`, not a push API |
| `trigger_flush` | `triggerFlush` | identical (`termSuffixes.size() >= chunkSize \|\| pendingDocs.size() >= maxDocsPerChunk`) |
| `flush(force)` | `flush(boolean force)` | identical, incl. the `(chunkDocs << 1) \| dirtyBit` token and the dirty accounting |
| `flush_num_fields` | `flushNumFields` | identical (the `chunkDocs == 1` vint special case included) |
| `flush_field_nums` | `flushFieldNums` | identical (sorted distinct field numbers, `(min(n-1,7) << 5) \| bitsRequired` token, overflow vint, headerless `PackedInts.PACKED`) |
| `flush_fields` | `flushFields` | identical (`DirectWriter` indices into the sorted array, `Arrays.binarySearch`) |
| `flush_flags` | `flushFlags` | identical, **both** selectors |
| `flush_num_terms` | `flushNumTerms` | identical |
| `flush_term_lengths` | `flushTermLengths` | identical (prefix stream then suffix stream) |
| `flush_term_freqs` | `flushTermFreqs` | identical |
| `flush_positions` | `flushPositions` | identical |
| `flush_offsets` | `flushOffsets` | identical arithmetic; one deliberate gating difference, finding 12 |
| `flush_payload_lengths` | `flushPayloadLengths` | identical |
| `common_prefix_len` | `StringHelper.bytesDifference` | identical result; no "terms out of order" throw (finding 4) |
| `copy_chunks` | `copyChunks` | equivalent; the leading/trailing partial-chunk loops are expressed as the chunk-boundary condition Java's `isLoaded(docID)` is really testing, exactly as `stored_fields::copy_chunks` does |
| `too_dirty` | `tooDirty` | identical |
| `can_bulk_copy` | `canPerformBulkMerge`'s compressor/chunkSize/version/packedIntsVersion/dirtiness half | equivalent (see finding 8) |
| `finish` | `finish(int numDocs)` | identical, incl. the final forced flush |
| `write_best_speed` | *(no counterpart)* | convenience wrapper over the writer, kept for every existing caller |
| `validate_field` | `addPosition`'s asserts | stricter (panics rather than silently emitting a desynchronised chunk) |
| *(no counterpart)* | `addProx` | not-in-Java-shape: this port never has a serialized prox stream to re-parse |
| *(no counterpart)* | `ramBytesUsed`/`getChildResources`/`close` | JVM accounting/lifecycle glue |

### Method correspondence (read side, changed only)

| Rust | Java | Verdict |
|---|---|---|
| `read_chunk` → `DecodedChunk` | `get(int doc)`'s metadata + decompression half | equivalent; decodes the whole chunk rather than one document's slice — finding 10 |
| `DecodedChunk::document` | `get(int doc)`'s `TVFields` materialisation | identical |
| `ChunkCursor` | *(no counterpart)* | Rust-only: Java's reader caches nothing between `get` calls |
| `chunk_for_doc` | `getIndexReader().getStartPointer(docID)` + the chunk's `docBase` | identical |
| `check_integrity` | `checkIntegrity()` (`CodecUtil.checksumEntireFile`) | identical — finding 3 |
| `chunk_size`/`num_chunks`/`num_dirty_chunks`/`num_dirty_docs`/`max_pointer`/`tvd` | `getChunkSize`/`getNumChunks`/`getNumDirtyChunks`/`getNumDirtyDocs`/`getMaxPointer`/`getVectorsStream` | identical — finding 9 |

### Findings

1. **[MISSING → fixed]** *The writer had no chunking at all — the c4 blocker.*
   Java's `finishDocument` calls `triggerFlush()` and closes the chunk at
   4 096 buffered term-suffix + payload bytes or 128 documents; the port set
   `chunk_docs = docs.len()`, `docBase = 0`, one non-dirty chunk, for the whole
   segment. Consequences, all three real: (a) no merge fast path could exist,
   because `copyChunks` needs a chunk-appending writer and a multi-chunk
   source; (b) fetching one document's vectors inflated a compression unit
   sized for the entire segment; (c) `numDirtyChunks`/`numDirtyDocs` were
   always `0`, so `tooDirty` — the safety switch that stops a degraded segment
   being copied forward forever — could never fire.
   *Resolution*: `TermVectorsWriter` is the streaming object Java's writer is
   (`pendingDocs`/`termSuffixes`/`numDocs`/`numChunks`/`numDirtyChunks`/
   `numDirtyDocs`, `triggerFlush`, `flush(force)`, `finish`), with the nine
   `flush*` header writers as free functions taking the pending-chunk slice.
   `write_best_speed` is a four-line wrapper over it. The per-field pending
   state (`PendingField`) is Java's `FieldData`, with the per-occurrence arrays
   owned per field rather than sliced out of four writer-wide scratch buffers
   by `posStart`/`offStart`/`payStart` — those buffers exist to dodge
   allocation for at most 128 documents, which buys nothing here and is where
   all of `addDocData`/`addField`'s offset arithmetic lives.
   Tests: `the_document_count_trigger_closes_a_chunk_at_128_documents`
   (asserts the exact `(docBase, chunkDocs, dirty)` header triple of all three
   chunks of a 300-document segment),
   `the_byte_size_trigger_closes_a_chunk_before_128_documents` (600-byte
   terms: the 4 096-byte trigger fires on the 7th document),
   `a_document_set_ending_exactly_on_a_chunk_boundary_still_flushes_a_dirty_tail`
   (128 documents ⇒ one clean chunk, zero dirty),
   `a_multi_chunk_segment_round_trips_positions_offsets_and_payloads`.

2. **[MISSING → fixed]** *`Lucene90CompressingTermVectorsWriter.merge`'s bulk
   path was absent* (c4's finding 13). Ported as
   `TermVectorsWriter::copy_chunks` plus `merge.rs`'s
   `write_merged_term_vectors`, following c4's stored-fields shape exactly:
   `MatchingReaders` gate, `liveDocs == null` gate, `can_bulk_copy`
   (compressor/chunk geometry/dirtiness), run detection over `doc_order`
   (Java's `while ((sub = docIDMerger.next()) == current)`), Java's two
   `CorruptIndexException` guards (`base != docID`, `docID > toDocID`) plus a
   `chunkDocs <= 0` guard, and the `(chunkDocs << 1) | dirty` token rebase
   (stored fields' is `<< 2`, which is the only format difference).
   Tests: `copy_chunks_of_two_whole_segments_reproduces_every_document`,
   `copy_chunks_of_a_partial_range_copies_the_ragged_ends_document_at_a_time`,
   `copy_chunks_of_a_range_inside_one_chunk_copies_no_chunk_at_all`,
   `copy_chunks_after_buffered_documents_forces_a_dirty_flush_first`,
   `copy_chunks_of_an_empty_range_writes_nothing`,
   `copy_chunks_rejects_an_out_of_range_or_inverted_document_range`,
   `a_chunk_header_whose_doc_base_disagrees_with_the_index_is_rejected`,
   `a_chunk_header_claiming_no_documents_is_rejected`,
   `a_chunk_claiming_more_documents_than_the_requested_range_is_rejected`,
   `dirtiness_accumulates_across_bulk_copies_until_the_segment_is_too_dirty`
   (130 repeated bulk copies build a segment that then fails `can_bulk_copy`,
   and still reads back correctly); and, at the merge layer,
   `matching_deletion_free_term_vector_sources_are_bulk_copied_verbatim`,
   `a_term_vector_source_with_deletions_is_re_encoded_not_copied`,
   `a_renumbered_term_vector_source_is_re_encoded_with_remapped_field_numbers`.

3. **[CORRECTNESS → fixed]** *No `checkIntegrity` before a byte copy.* Java's
   `merge` runs `reader.checkIntegrity(mergeState.oneMerge)` on every source
   *before* it picks a strategy. `term_vectors::open` only calls
   `retrieve_checksum`, which validates the footer's **shape** and not the CRC.
   That is the right trade for a random-access reader and fatal for a byte
   copy: the bulk path copies a source's compressed bytes verbatim and then
   writes a freshly computed, valid footer over them, so a corrupt source would
   become a merged segment that passes every checksum from then on — the hazard
   behind Java's "bulk merge is scary: its caused corruption bugs in the past"
   comment, and the same defect c4 found on the stored-fields side (its
   finding 17).
   *Resolution*: `TermVectorsReader::check_integrity()`
   (`codec_util::check_whole_file_footer` over `tvd[..max_pointer]`), called on
   every source in `write_merged_term_vectors` at exactly the point Java calls
   it. Tests: `check_integrity_detects_payload_corruption_that_the_footer_shape_cannot`
   and, end to end,
   `a_term_vector_source_whose_tvd_fails_its_own_checksum_is_never_bulk_copied`
   — both flip one byte of a chunk body, leaving every length, pointer and
   footer field intact, i.e. precisely the corruption `retrieve_checksum`
   cannot see.

4. **[MISSING → fixed]** *No term prefix sharing.* Java's `startTerm` computes
   `StringHelper.bytesDifference(lastTerm, term)` — `lastTerm` reset at
   `startField` — and writes only the suffix; the port wrote `prefix_len = 0`
   for every term. Not a correctness bug (the reader rebuilds
   `lastTerm[..prefix] + suffix` either way, and `prefix = 0` is a legal
   encoding) but it inflates the LZ4 unit, and — because `termSuffixes.size()`
   *is* the chunk-size trigger — it also changed where chunks close.
   *Resolution*: `common_prefix_len`, applied per field. Java throws
   `IllegalArgumentException` on two equal consecutive terms
   (`Arrays.mismatch` returning `-1`); this port shares the whole length
   instead, because Java can afford the throw only by virtue of being fed a
   `TermsEnum`, while `TermVectorsDocument` is a plain value a caller builds.
   Test: `terms_are_prefix_compressed_against_the_previous_term_in_the_field`
   compares two documents with the same total term bytes, one where
   consecutive terms share a long prefix and one where they share nothing, and
   asserts the first `.tvd` is strictly smaller — an assertion only prefix
   sharing can satisfy.

5. **[MISSING → fixed]** *`charsPerTerm` was a hard-coded `1.0`.* Java's
   `flushOffsets` derives it per distinct field number as
   `sumOffsets / sumPos`, both summed over each term's **last** occurrence, and
   subtracts `(int)(cpt * positionDelta)` from every start-offset delta; the
   read side adds exactly the same quantity back. A constant is therefore
   round-trip-correct (b7 recorded it as intentional) but makes the delta
   stream far less compressible: with the real ratio, a uniformly tokenized
   field encodes every start-offset delta as **0**.
   *Resolution*: `flush_offsets` is now Java's computation, including the
   `(sumPos <= 0 || sumOffsets <= 0) ? 0` guard for a field with offsets but no
   positions. Tests: `chars_per_term_is_the_offset_to_position_ratio_of_the_field`
   (asserts both the emitted `6.0` and that all three encoded deltas are 0),
   `a_field_with_offsets_but_no_positions_gets_a_zero_chars_per_term`.

6. **[MISSING → fixed]** *`flushFlags`' `nonChangingFlags` encoding was never
   emitted.* Java writes selector `0` — one 4-bit flag per *distinct field
   number* — whenever every instance of a field number agrees, and only falls
   back to selector `1` (one per field instance) otherwise. The port always
   wrote selector `1`. Both decode (the reader has always had both branches),
   but a 128-document chunk of two fields cost 128 bytes of flags where Java
   costs one. Only reachable at all now that chunks hold more than a handful
   of documents.
   *Resolution*: `flush_flags` implements Java's `outer:`-labelled agreement
   scan and both branches. Test:
   `flags_are_written_once_per_field_number_when_they_never_change`.

7. **[MISSING → fixed]** *Distinct field numbers were written in first-seen
   order; `.tvx` used `blockShift = 0`.* `flushFieldNums` sorts (and
   `flushFields`/`flushFlags`/`flushOffsets` then `Arrays.binarySearch` into
   the sorted array); the format's `blockShift` is `10`, not `0`. Neither was a
   correctness bug — the reader resolves fields through `fieldNumOffs` in any
   order, and a `DirectMonotonicWriter` with `blockShift = 0` is one block per
   value — but the second is now load-bearing: a single-chunk segment had two
   `.tvx` entries and a 57-chunk one has 58, where one block per value is the
   worst possible packing.
   *Resolution*: both are Java's now. Test:
   `distinct_field_numbers_are_written_sorted` (also asserts the *document's*
   field order is preserved regardless).

8. **[MISSING → fixed]** *No `chunkSize`/`maxDocsPerChunk` writer parameters.*
   `Lucene90CompressingTermVectorsWriter`'s constructor takes them;
   `Lucene90TermVectorsFormat` fixes them at `1 << 12` / `128`. The port
   hard-coded the format's values, which left `can_bulk_copy`'s
   `reader.getChunkSize() == chunkSize` comparison — the one that stops a
   segment written under a different geometry from being spliced into this
   one — comparing against a constant it could never differ from.
   *Resolution*: `TermVectorsWriter::with_geometry`, with `new` delegating to
   it; `chunk_size`/`max_docs_per_chunk` are fields. Both are asserted to fit
   in `i32` (`.tvm` records `chunkSize` as a vint). This is also what lets the
   benchmark reproduce the pre-`c8` writer exactly rather than approximate it.
   Test: `a_custom_chunk_geometry_is_recorded_and_blocks_a_default_geometry_bulk_copy`.

9. **[MISSING → fixed]** *The reader discarded four `.tvm` fields it had
   parsed.* `chunkSize` was read into `_chunk_size`; `maxPointer`,
   `numDirtyChunks` and `numDirtyDocs` were read, validated and dropped. Java
   exposes all four (`getChunkSize`/`getMaxPointer`/`getNumDirtyChunks`/
   `getNumDirtyDocs`) and `canPerformBulkMerge`/`tooDirty`/`copyChunks` need
   every one of them.
   *Resolution*: stored and exposed, plus `chunk_for_doc` and `tvd()`.

10. **[PERF → fixed]** *Every `document()` call re-decoded a whole chunk.*
    b7's finding 20, now bounded by the chunk instead of the segment — but the
    merge's per-document path walks 128 consecutive documents out of the same
    chunk, and decoding it 128 times is the difference between O(documents)
    and O(chunks) decodes. Java pays this too (its reader caches nothing
    between `get` calls), so this is a place the port can be *faster* than
    Lucene rather than merely equal.
    *Resolution*: `read_chunk` returns a `DecodedChunk` holding every packed
    array and the decompressed LZ4 unit; `DecodedChunk::document` materialises
    one document out of it; `ChunkCursor` reloads only when the requested
    document falls outside the chunk it holds, and is what
    `write_merged_term_vectors`' per-document path and `copy_chunks`' two
    ragged-end loops use. Tests:
    `a_chunk_cursor_serves_every_document_of_a_chunk_from_one_decode`
    (byte-equality with the random-access read for all 300 documents, forwards
    and backwards, so a stale chunk fails),
    `read_chunk_reports_its_own_extent_and_rejects_documents_outside_it`.

11. **[CORRECTNESS → fixed]** *An empty document set wrote a phantom chunk.*
    `write_best_speed(&[])` emitted `docBase = 0`, a `chunkDocs = 0` token, and
    recorded `numChunks = 1`. Java's `finish(0)` finds `pendingDocs` empty,
    never flushes, and records `numChunks = 0`. Benign in practice (`maxDoc` is
    0, so no reader ever seeks into it) but it is a byte-level divergence in a
    header field, and the new writer's `flush` asserts `chunkDocs > 0` exactly
    as Java does. Now `numChunks = 0`. Covered by the pre-existing
    `write_best_speed_empty_doc_set_produces_zero_max_doc`, which passes
    unchanged.

12. **[CORRECTNESS, preserved deliberately]** *b7's `totalOffsets`/
    `totalPayloads` gating is intact and is now documented as a divergence
    rather than a coincidence.* Java's `flushOffsets` returns early on
    `!hasOffsets`, i.e. "no field carries the OFFSETS flag"; both readers —
    Java's `get` and this port's `read_chunk` — instead decide the streams are
    present from `totalOffsets > 0`, a sum of term *frequencies*. The two
    conditions differ for exactly one input Java's own writer can never produce
    and this port's public `TermVectorsDocument` can: a field carrying OFFSETS
    with **no terms**. The rewrite keeps b7's gating (`total_offsets == 0`
    returns early) and says why in the code.
    Regression test `write_best_speed_flagged_but_termless_field_round_trips`
    passes unchanged.

13. **[INTENTIONAL]** Remaining scope cuts, all compression-ratio-only: the LZ4
    unit is still a single literal run (no match finding), and every
    `direct_reader`/`packed_ints`/`block_packed` array still uses the exact bit
    width its own values need rather than a cross-chunk minimisation. Both
    produce bytes real Lucene reads (finding 16).

### Verdict

Swept clean on the write side, which is now a full port of
`Lucene90CompressingTermVectorsWriter` rather than a scoped-down encoder:
chunking, both flush triggers, all nine per-chunk header writers with both
flags encodings, prefix compression, the derived `charsPerTerm`, the
dirty-chunk accounting, `tooDirty`, and `copyChunks`. One CORRECTNESS defect
found and fixed on the way (finding 3, the missing `checkIntegrity`) and one
byte-level divergence (finding 11). Remaining open items are the two
compression-ratio scope cuts in finding 13.

---

## `crates/lucene-codecs/src/postings.rs` (+ `for_util.rs`, `blocktree.rs`)

Java counterparts:
- `lucene/core/src/java/org/apache/lucene/codecs/lucene104/Lucene104PostingsReader.java`
- `lucene/core/src/java/org/apache/lucene/codecs/lucene104/PForUtil.java`
- `lucene/core/src/java/org/apache/lucene/codecs/lucene104/PostingsUtil.java`
- `lucene/core/src/java/org/apache/lucene/index/PostingsEnum.java`

| Rust | Java | Verdict |
|---|---|---|
| `PostingsFlags` | `PostingsEnum`'s `NONE`/`DOCS`/`FREQS`/`POSITIONS`/`OFFSETS`/`PAYLOADS`/`ALL` + `featureRequested` | reduced deliberately — finding 14 |
| `DocInput::read_postings_with_flags` / `lazy_cursor_with_flags` | `Lucene104PostingsReader.postings(fieldInfo, termState, reuse, flags)` | equivalent |
| `for_util::pfor_skip` | `PForUtil.skip` | identical |
| `decode_full_block_body`'s `needs_freq` branch | `refillFullBlock`'s `if (indexHasFreq) { if (needsFreq) freqFP = …; PForUtil.skip(docIn); }` | equivalent — finding 15 |
| `read_tail_block`'s `needs_freq` branch | `PostingsUtil.readVIntBlock(…, indexHasFreq, decodeFreq)` | identical, all three branches |
| `blocktree::FieldTerms::postings_with_flags` / `lazy_postings_with_flags` | `TermsEnum.postings(reuse, flags)` | equivalent |
| `check_wire_position` | `assert docIn.getFilePointer() == blockEndFP` / `== skip1EndFP` | stricter — finding 16 |

### Findings

14. **[PERF → fixed]** *b5's F6: freqs were always decoded.* Java separates
    `indexHasFreq` (the field's index options) from
    `needsFreq = indexHasFreq && PostingsEnum.featureRequested(flags, FREQS)`.
    When the field has frequencies and the consumer does not want them,
    `refillFullBlock` records `freqFP` and calls `PForUtil.skip(docIn)` — one
    token byte and a seek — instead of a 256-value unpack, and
    `PostingsUtil.readVIntBlock` skips the tail block's trailing
    freq-exception vints. This port always ran `pfor_decode` and always read
    the exception vints.
    *Resolution*: `PostingsFlags` (`DocsOnly`/`Freqs`), threaded through
    `read_postings_with_flags`/`lazy_cursor_with_flags` and
    `blocktree::FieldTerms::{postings_with_flags, lazy_postings_with_flags}`;
    the original four entry points remain, delegating with `Freqs`, so no
    caller in a crate this batch does not own had to change. New
    `for_util::pfor_skip`.
    Rather than Java's six-constant mask, the type is the **one** distinction
    `Lucene104PostingsReader` derives from that mask before it touches `.doc` —
    the other flags gate `.pos`/`.pay`, which in this port are a separate call
    (`read_positions`) a docs-only consumer simply does not make. Java's own
    chain (`POSITIONS` subsumes `FREQS`, `OFFSETS`/`PAYLOADS` subsume
    `POSITIONS`, `NONE == DOCS`) is what makes that reduction lossless, and the
    type's doc comment states it.
    Where Java defers via `freqFP` and decodes lazily if `freq()` is actually
    called, this port makes "not requested" mean "not available": every
    frequency reads back as `1`. That is exactly the `PostingsEnum.NONE`/`DOCS`
    contract, and it avoids carrying a second file position and a
    seek-and-decode path through the cursor for a case whose whole point is not
    to do that work.
    Tests: `for_util::pfor_skip_consumes_exactly_what_pfor_decode_does` (five
    block shapes — all-equal, all-equal-with-exceptions, packed,
    packed-with-exceptions, maximum width — each asserting `skip` lands on the
    same byte `decode` does, with a trailing sentinel so an overshoot is
    visible), `pfor_skip_reports_eof_rather_than_running_off_the_end`,
    `postings_writer::docs_only_flags_decode_the_same_doc_ids_and_report_freq_one`
    (three `docFreq` shapes at once — full blocks + tail, full blocks only, a
    whole level-1 span + tail — through the eager path, an exhaustive lazy
    walk and a skip-heavy `advance`, with an assertion that the fixture really
    does have non-uniform frequencies so the test cannot pass vacuously), and
    `docs_only_flags_are_a_no_op_for_a_field_without_frequencies`.
    *Wired*: `term_delete::resolve_term_doc_ids` — delete-by-term resolution
    reads doc ids and never looks at a frequency. *Left open*: the
    constant-score/filter/`TermInSetQuery` call sites in `lucene-search`; that
    crate was under concurrent edit by `c11` for the whole of this batch and is
    not in this batch's gate, so flipping call sites there could not have been
    verified. The API they need exists and is tested.

15. **[PERF, recorded]** *Level-1 impacts are still decoded for skipped spans*
    (b5's F7), and `read_postings`' eager path still decodes level-0 impacts
    even for a `DocsOnly` request. Both are bounded — one level-1 entry per
    8 192 documents, and the level-0 impact bytes are a borrow decoded only
    when asked — and Java gates the second on `needsImpacts`, which is a
    separate flag from `PostingsEnum`'s mask (it comes from which enum class
    `Lucene104PostingsReader.impacts` constructs). Threading it is a second
    axis this batch did not need; recorded rather than half-done.

16. **[CORRECTNESS → fixed]** *A corrupt `.doc` panicked in a debug build
    where every other wire-level disagreement in the module returns an error.*
    Raised independently by `c9`, whose byte-flipping `check_index` sweep hit
    it and had to wrap `check_directory` in `catch_unwind` with a silenced
    panic hook to get past it.
    The `.doc` stream records two byte lengths **redundantly**, i.e. as values
    read off the same file as the data they measure: a level-0 header's
    `level0NumBytes` (the body that follows) and a level-1 entry's
    `skip1EndFP` (that entry's own metadata). Neither is derivable from what it
    measures, so on a corrupt file the decode ends somewhere else and every
    byte after it is garbage. Java asserts both; this port used
    `debug_assert_eq!` in four places — both loops of `read_postings`,
    `LazyDocsCursor::refill`, and `read_level1_entry` — which is a panic in
    debug and a *silent accept* in release. In a debug build of the FFI that
    panic takes the JVM down. `debug_assert` is for invariants this code's own
    arithmetic guarantees; these are bytes off disk. This is the same class
    b5's F2 already fixed in `advance`.
    *Resolution*: `check_wire_position(position, expected, what)` returns
    `Error::Store(Corrupted)` from all four sites. Tests:
    `a_block_body_that_disagrees_with_its_headers_byte_length_is_rejected`
    (a level-0 header claiming one byte more than its body occupies; both the
    eager and the lazy path must return `Corrupted`) and
    `a_level1_entry_that_disagrees_with_its_own_skip1_end_fp_is_rejected`
    (which first asserts the *honest* entry does **not** trip the check, so
    the test cannot pass for the wrong reason, then lies by one byte). Neither
    needs `catch_unwind`.
    The rest of `postings.rs`'s non-test assertions were swept for the same
    pattern: `read_tail_block`'s `debug_assert_eq!(freqs.len(), count)` and the
    four `.expect(...)`s in the position/payload readers are all guaranteed by
    this code's own control flow, not by file contents, and are correct as
    they stand.
    **Note for `c9`**: the `catch_unwind`/`set_hook` scaffolding in
    `corrupting_the_doc_skip_data_is_caught_by_the_advance_check`, and the
    cross-batch finding it records, can both be dropped — that panic no longer
    exists, and the corruption now surfaces as a normal `check_directory`
    failure.

17. **[PERF, assessed and recorded — not done]** *`FieldPostingsInput` carries
    no norms, so every impact is `(maxFreq, 1)`.* b5's F12. The brief asked
    whether b13's `SegmentReader::field_norms()` makes threading norms cheap
    now. It does not, for three separate reasons, and the assessment is worth
    recording so the next batch does not re-derive it:

    - **`field_norms()` is the wrong side of the format.** It hands out a
      `FieldNorms` over an *already-written* segment's `.nvm`/`.nvd`, for
      scoring. `Lucene104PostingsWriter.startTerm(NumericDocValues norms)`
      needs the norms of the documents it is *about to write*, before any
      `.nvd` for that segment exists. On the flush path the values do exist in
      memory (`index_writer`'s norms output is built from the same
      `invert_documents` pass as the postings), so the plumbing is
      `FieldPostingsInput { norms: Option<&[i64]> }` plus a call-site change —
      but that call site is `index_writer.rs`, which `c7` held for the whole
      of this batch.
    - **On the merge path there are no norms to thread at all.** c4's own
      still-open carry-over is that `IndexWriter::execute_merge` passes
      `norms: &[]`, so a merged segment has none; `merge_postings` could not
      supply what the merge never opened. That carry-over has to close first.
    - **Real norms make the current one-impact-per-block encoding wrong, not
      merely loose.** With a constant norm, Java's
      `CompetitiveImpactAccumulator.getCompetitiveFreqNormPairs` collapses to
      the single highest-freq entry — which is exactly what
      `write_level1_span` computes, and why the current output *is* Java's
      output for a norm-less field rather than an approximation of it. With
      varying norms the correct output is the competitive frontier over
      `(freq, norm)`, so `CompetitiveImpactAccumulator` has to be ported in the
      same change or the emitted bounds become unsound rather than loose.

    Until then the current bound is sound (norm 1 is the shortest field
    length, hence the highest possible score), so it costs pruning
    opportunities and never drops a hit. Recorded, unchanged, with the three
    prerequisites named.

### Verdict

b5's F6 closed. b5's F7 (15) and F12 (17) stay open by choice, the second
with its three prerequisites named. One CORRECTNESS defect fixed (16) that was
blocking another batch's corruption sweep.

---

## `crates/lucene-index/src/merge.rs` (term-vectors merge only)

Java counterparts: `Lucene90CompressingTermVectorsWriter.merge`/
`canPerformBulkMerge`/`copyChunks`, `codecs/compressing/MatchingReaders`.

| Rust | Java | Verdict |
|---|---|---|
| `write_merged_term_vectors` | `merge(MergeState)`'s `DocIDMerger` loop | equivalent (the loop walks the `doc_order` both entry points already compute) |
| `term_vectors_merge_strategy` | `canPerformBulkMerge` | equivalent (`matching_readers` + `liveDocs == null` + `can_bulk_copy`) |
| `matching_readers` | `MatchingReaders` | reused unchanged from c4 |
| *(no counterpart)* | `BULK_MERGE_ENABLED` system property | not ported — no system-property mechanism here; the equivalent is deleting the `Bulk` arm |

### Findings

18. **[PERF → fixed]** *c4's finding 13: the term-vectors merge materialised
    and re-encoded every document.* `merge_term_vectors` built a
    `Vec<TermVectorsDocument>` of the whole merged segment — one owned `Vec` per
    field per document, each holding owned term bytes, positions, offsets and
    payloads — and handed it to a single-chunk `write_best_speed`. On a
    single-chunk source, each `reader.document(doc)` decoded the *entire*
    source segment, so the read half alone was O(documents²).
    *Resolution*: `write_merged_term_vectors` picks a strategy per source and
    streams. A matching, deletion-free source with the same chunk geometry has
    its compressed chunks copied byte for byte; every other source is decoded
    through a per-source `ChunkCursor` and re-encoded with its field numbers
    remapped. A source with no term-vectors reader at all still contributes an
    empty document per doc, exactly as `TermVectorsWriter.merge` does for a
    null `mergeState.termVectorsReaders[i]` — covered by the pre-existing
    `term_vectors_merge_across_two_sources_with_deletions_and_a_source_with_none`,
    which passes unchanged.

19. **[CORRECTNESS → fixed]** The `checkIntegrity`-before-byte-copy fix,
    finding 3, lives here: `write_merged_term_vectors` calls
    `reader.check_integrity()` on every source before any strategy is chosen.

20. **[PERF → fixed]** *The per-document merge path decoded a chunk per
    document.* Fixed by finding 10's `ChunkCursor`, one per source.

21. **[MISSING → fixed]** *`write_merged_segment_fixture` no longer compiled.*
    `c7` renamed `IndexWriter::delete_documents` to
    `delete_documents_with_sources`, which broke `c4`'s example — and with it
    `cargo test -p lucene-index` and the whole of `scripts/verify-write-path.sh`.
    Repaired in place (one call site).

### Verdict

c4's last carry-over is closed. `merge.rs` edits are confined to the
term-vectors merge and the two helper items above; the stored-fields, postings
and points merges are untouched.

---

## Verification: real Lucene reads the chunked bytes

Chunking changes the bytes on disk, so the read-side floor and the write-side
proof both had to move.

**Read-side floor, unchanged.** `crates/lucene-codecs/tests/term_vectors_fixtures.rs`
(`parses_real_term_vectors_and_matches_lucene_positions_offsets_payloads`) reads
real `IndexWriter`-produced `.tvd`/`.tvx`/`.tvm` bytes and passes untouched, as
do all of `term_vectors.rs`'s pre-existing unit tests.

**`VerifyTermVectors` gained a third segment.**
`write_term_vectors_fixture` now writes `_2`: **400 documents** across a
dozen-odd chunks, two fields per document with constant flags (so `flushFlags`
takes the per-field-number branch), the first field carrying positions,
offsets **and** payloads (so `flushOffsets`' derived `charsPerTerm` and
`flushPayloadLengths` are both live), and consecutive terms sharing a long
prefix (so `startTerm`'s compression is). The Java verifier renders every
occurrence's position, start offset, end offset and payload bytes through real
`PostingsEnum.ALL` and compares against a manifest the Rust side wrote
independently. This is the only case that proves real Lucene can locate a
document in a chunk *other than the first* through `.tvx`'s
`DirectMonotonicReader`. It is a live check, not a rubber stamp: it failed on
its first run on a genuine rendering difference (a no-offsets field reports
`-1`, not "absent") before passing.

**`VerifyMergedSegment` now covers a bulk-merged term-vector segment.**
`write_merged_segment_fixture`'s indexed field stores term vectors, so the
merge of its three 2 400-document segments exercises both term-vector merge
strategies in one run: `_0` and `_2` are matching and deletion-free (BULK),
`_1` has real `.liv` deletions (per-document). Term-vector chunks are 4 096
bytes / 128 documents against stored fields' 80 kB / 1 024, so the copy loop
runs over an order of magnitude more boundaries than the stored-fields one.
The verifier recomputes every one of the 7 176 merged documents' expected
vectors from that document's own stored `body` text and compares term by term
with frequencies, then runs real `CheckIndex` at
`MIN_LEVEL_FOR_SLOW_CHECKS`. Confirmed the bulk path actually fired: the merged
`.tvm` records 57 chunks with **3** dirty ones and 264 dirty docs — the two
bulk-copied sources' own dirty tails carried across verbatim plus the merge's
own final flush, where a fully re-encoding merge would record exactly one.

---

## Gate

- `cargo fmt --all` — clean.
- `cargo clippy -p lucene-codecs -p lucene-index --all-targets -- -D warnings` — clean.
- `cargo test -p lucene-codecs -p lucene-index` — green, whole suite, no
  skips: **1 064** `lucene-codecs` lib tests, **515** `lucene-index` lib tests,
  every integration target, 0 failures. Worth naming: `c9`'s new
  `corrupting_the_doc_skip_data_is_caught_by_the_advance_check` took ~10
  CPU-minutes for most of this batch, unwinding through the panic finding 16
  removed; with that fix it **passes in 0.36 s**.
- `scripts/verify-write-path.sh` — **18/18**. (It was 16/16 at the start of the
  batch; `c5` and `c7` added a vector-segment and a block-segment case
  alongside.) Includes this batch's extended `VerifyTermVectors` — three
  segments, one of them multi-chunk with offsets and payloads — and
  `VerifyMergedSegment` with a bulk-merged term-vector segment plus real
  `CheckIndex`.
- `cargo llvm-cov -p lucene-codecs -p lucene-index --summary-only`, lines:
  `term_vectors.rs` **98.49 %**, `merge.rs` **98.72 %**, `for_util.rs`
  **98.37 %**, `term_delete.rs` **99.19 %**, `postings_writer.rs` **99.53 %** —
  all above the 95 % bar. `postings.rs` is **89.79 %**, which is *not* a
  regression this batch introduced: the uncovered block is lines ~1035–1294,
  the `read_positions`/`read_positions_flat`/`read_positions_for_docs` family,
  which is only ever called from `lucene-search` and so is invisible to a
  two-crate coverage run. Every line this batch added to that file has a
  direct test.

---

## Carry-over items raised by this batch

- [ ] **`PostingsFlags::DocsOnly` is not wired into `lucene-search`.** The
      constant-score / filter / `TermInSetQuery` / doc-only scorer paths named
      in this batch's brief all still ask for frequencies. `lucene-search` was
      under concurrent edit by `c11` throughout this batch and is not in this
      batch's gate, so flipping call sites there could not have been verified;
      the API (`FieldTerms::{postings_with_flags, lazy_postings_with_flags}`)
      exists, is tested, and is already used by
      `term_delete::resolve_term_doc_ids`. Owner: `c11` or a successor.
- [ ] **`FieldPostingsInput` still carries no norms**, so impacts stay
      `(maxFreq, 1)` — which *is* Java's output for a norm-less field, and a
      sound-but-loose bound otherwise. Finding 17 records the assessment the
      brief asked for: b13's `SegmentReader::field_norms()` does not help (it
      is the read side), the merge path has no norms to thread until c4's
      `execute_merge` carry-over closes, and real norms require
      `CompetitiveImpactAccumulator` in the same change or the emitted bounds
      become unsound rather than loose.
- [ ] **`needsImpacts` is a second flag axis, unported.** `read_postings`
      decodes level-0 impacts even for a `DocsOnly` request, and level-1
      impacts are decoded for spans being skipped (b5's F7). Both bounded;
      see finding 15.
- [ ] **The LZ4 unit is still a single literal run** and every packed array
      still uses its own exact bit width. Compression ratio only; real Lucene
      reads the output either way. (Finding 13.)
- [x] **`c9` can drop its `catch_unwind` scaffolding.** Finding 16 removed the
      panic that
      `check_index::corrupting_the_doc_skip_data_is_caught_by_the_advance_check`
      works around; a corrupt `.doc` block or level-1 entry now returns
      `Corrupted`. Handed back to `c9`.

## Concurrency

`term_vectors.rs` and `postings.rs` were this batch's. `blocktree.rs` and
`for_util.rs` received additive functions only. `merge.rs`'s edits are confined
to the term-vectors merge. Three files owned by other in-flight batches were
touched minimally and deliberately: three doc-comment references to the
renamed `merge_term_vectors` in `index_writer.rs` (`c7`), one call site in
`term_delete.rs` (`c7`'s crate), and the repair of `c7`'s `delete_documents`
rename in `c4`'s `write_merged_segment_fixture.rs` (finding 20) — without which
`cargo test -p lucene-index` and all of `verify-write-path.sh` were broken for
everyone.

The workspace was broken by other batches several times mid-run: `check_index.rs`
mid-refactor (`c9`) for the first hour, `lucene-ffi`'s `Term` import (`c7`),
and `BooleanQuery`'s new `filter` field (`c11`). Per protocol this batch waited
and retried rather than editing any of them.

---

## Summary of findings

**20 distinct findings** across 21 numbered entries (3 and 19 are the same
`checkIntegrity` fix, recorded once per file it spans).

| Class | Count | Findings |
|---|---|---|
| `CORRECTNESS` | 4 | 3 / 19 (no `checkIntegrity` before a byte copy), 11 (phantom empty chunk), 12 (b7's gating preserved and documented), 16 (a corrupt `.doc` panicked in debug) — all fixed |
| `MISSING` | 9 | 1, 2, 4, 5, 6, 7, 8, 9, 21 — all fixed |
| `PERF` | 6 | 10, 14, 18, 20 fixed and measured; 15 and 17 assessed and recorded |
| `INTENTIONAL` | 1 | 13 |

Both briefed carry-overs are closed: c4's finding 13 (the term-vectors bulk
merge, which needed real chunking first) and b5's F6 (`PostingsEnum` flags).
Four CORRECTNESS defects were fixed on the way, two of them reachable from a
corrupt file on disk.
