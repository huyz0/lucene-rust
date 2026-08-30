# c15 — postings API: offsets for one document, the `.dvs` check, and norms field validation

Follow-up batch. Three carry-overs, each blocked on a file that is now free,
plus a full re-audit of `postings.rs` for the panic/abort class that has now
produced five separate defects in that one file.

Files swept (owned by this batch):

Java read from **`/home/tuong/work/lucene-10.5.0`**, the pinned tag, not
`/home/tuong/work/lucene` (which is `main`, thousands of commits ahead) — see
c18's handoff and `PROTOCOL.md`. All three classes were diffed tag-vs-`main`
to be sure nothing this batch relied on is a `main`-ism:

| Rust file | Java counterpart (10.5.0) | tag vs `main` |
|---|---|---|
| `crates/lucene-codecs/src/postings.rs` | `lucene/core/src/java/org/apache/lucene/codecs/lucene104/Lucene104PostingsReader.java` (+ `PostingsUtil.java`, `PForUtil.java`) | differs only in `try`/`finally` resource handling and `checkIntegrity(MergePolicy.OneMerge)` — nothing this batch reads |
| `crates/lucene-codecs/src/norms.rs` | `lucene/core/src/java/org/apache/lucene/codecs/lucene90/Lucene90NormsProducer.java` | `readFields` is **identical**; the diff is resource handling, `prefetch`'s return type and `checkIntegrity` |
| `crates/lucene-index/src/check_index.rs` | `lucene/core/src/java/org/apache/lucene/index/CheckIndex.java` | `checkDocValueSkipper` is **byte-identical** in both. `testHnswGraphs` exists only on `main` — see F15 |

Files touched additively (not owned, edits kept minimal and mechanical — see
[Concurrency](#concurrency)): `lucene-codecs/src/blocktree.rs`,
`lucene-search/src/{highlighter.rs,lib.rs,directory_reader.rs}`,
`lucene-search/benches/highlight_offsets.rs` (new),
`lucene-codecs/tests/postings_wanted_docs.rs` (new).

Findings: **4 CORRECTNESS** (all fixed), **3 MISSING** (all fixed), **4 PERF**
(3 fixed and measured, 1 recorded with the blocker named), **1 INTENTIONAL**,
plus **2 documentation corrections** handed over by `c18-version-audit`.

---

## `crates/lucene-codecs/src/postings.rs`

Java: `Lucene104PostingsReader` (the `.doc`/`.pos`/`.pay` reader), plus
`PostingsUtil.readVIntBlock` and `PForUtil.skip`.

### Method correspondence (the functions this batch touched or added)

| Rust | Java | Verdict |
|---|---|---|
| `read_positions` | `BlockPostingsEnum.nextPosition`/`startOffset`/`endOffset`/`getPayload`, run over every doc | divergent by shape (materializes; Java streams) — pre-existing, unchanged |
| `read_positions_flat` | same, positions only | divergent by shape — pre-existing, unchanged |
| `read_positions_for_docs` | `advance(doc)` + `nextPosition()`, for a batch of docs | **reimplemented** on the shared walker (F1, F5, F6) |
| `read_occurrences_for_docs` | `advance(doc)` + `nextPosition()`/`startOffset()`/`endOffset()`/`getPayload()` | **added** (F1) |
| `walk_wanted_occurrences` / `OccurrenceSink` / `SinkCursor` / `wanted_ranges` / `skip_position_block` | *(no Java equivalent: Java's enum is a stateful cursor, not a batch)* | not-in-Java, intentional (F11) |
| `decode_position_streams` | `refillPositions` + `refillLastPositionBlock` | identical; allocation bounds hardened (F3) |
| `read_level1_entry` | `skipLevel1To`'s entry read | identical; `numImpactBytes` hardened (F2) |
| `read_full_block_header` | `doMoveToNextLevel0Block`/`skipLevel0To` | identical; impacts length + `blockLength` hardened (F2) |
| `read_postings_with_flags` | `BlockPostingsEnum` driven to exhaustion | divergent by shape (eager); `docFreq` reservation hardened (F3) |
| `wire_length` / `add_wire_offset` / `wire_count` / `corrupted` | *(Java relies on `NegativeArraySizeException`/`readBytes` throwing)* | not-in-Java, intentional |
| every position/offset/doc-id accumulator | Java accumulates in an `int` and wraps silently | now `wrapping_add` everywhere (F2) |
| `LazyDocsCursor::advance`'s tail-block guard | `refillRemainder`'s `assert docCountLeft >= 0 && < BLOCK_SIZE` | fixed by the main session just before this batch; **now has a test** (F4) |

Java methods with no Rust counterpart, unchanged from b5/c8: `impacts()`'s
separate `needsImpacts` axis, `postings()`'s `reuse` recycling, and the
`.pos`/`.pay` file pointers in the skip data (F10).

### F1 `[MISSING → fixed]` The offsets API forced a whole-postings-list decode

**Java**: `PostingsOffsetStrategy.getOffsetsEnum` does
`postingsEnum.advance(doc)` and then walks that one document's positions,
reading `startOffset()`/`endOffset()`/`getPayload()` as it goes.

**We did**: `blocktree::FieldTerms::positions` was the only offset-carrying
accessor in the port, and it returns every document's occurrences. The
highlighter therefore decoded the whole term's `.pos`/`.pay` and allocated one
`Vec<Position>` per document in the term — plus a `Vec<u8>` per occurrence —
to read one document's offsets. c12 §3.4 recorded it with the blocker named:
`read_positions_for_docs` exists for exactly this shape but returns positions
only.

c12's stated reason why the offsets could not be recovered — "the wire format
deltas offsets cumulatively within a document, so a skipping decoder would
have to carry per-document start-offset state it deliberately does not keep" —
**is not right**, and that is what makes the fix small. Both the position and
the start-offset accumulator reset at each document's *first* occurrence
(`Lucene104PostingsWriter.startDoc`'s `lastPosition = 0; lastStartOffset = 0`),
so a document's occurrences are entirely self-contained. Nothing carried
across the documents a walk skips is needed to decode the ones it keeps —
which is also why skipping is sound at all.

**Fixed**: `read_occurrences_for_docs`, and `read_positions_for_docs`
reimplemented on top of the same walker so there is one wire decoder rather
than two. The two differ only in an `OccurrenceSink` implementation, chosen at
the type level (`const NEEDS_EXTRAS`), so a positions-only walk never unpacks
a byte of `.pay` and a highlighting walk never pays for a per-document `Vec`
it does not want.

`blocktree::FieldTerms` gains `occurrences_for_docs` (the batch form) and
`occurrences_for_doc` (the `advance(doc)`-then-walk form the highlighter
wants; `Ok(None)` for a term the field lacks or a document the term is not
in). `highlighter::offsets_from_postings` is three calls shorter and one call
per term now.

Tests: `crates/lucene-codecs/tests/postings_wanted_docs.rs` — five tests, all
**differential against `FieldTerms::positions`**, the whole-term reader
`blocktree_fixtures.rs` already pins against real Lucene's own occurrence
list. Expectations are never hand-transcribed: they are sliced out of the
verified reader's output. `wanted_documents_agree_with_the_whole_term_reader`
covers four field shapes (positions; +payloads; +offsets; +both) × eight
`wanted` subsets over a 200-document / 1 000-occurrence term (three full
256-wide blocks and a 232-occurrence tail), and cross-checks
`positions_for_docs` and `positions_flat` against the same expectations.
`occurrences_for_doc_matches_the_whole_term_reader_for_every_document` walks
all 200 documents one at a time.

### F2 `[CORRECTNESS → fixed]` Lengths and accumulators read off disk overflowed

The class the brief asked for an audit of, and the one that has now produced
five defects in this file (b5 F2, c8 finding 16 ×4, the main session's
lazy-cursor tail block). A **length or count read off disk** must never reach
an index, slice, allocation or shift without validation; in a debug build the
result is a panic, and a panic in a debug build of the FFI takes the JVM down
with no exception for the caller to catch.

**Lengths**: three live sites, all of the same shape — a signed wire value,
`as usize` sign-extending it to ~2^64, and `base + len` overflowing:

| Site | Wire type | Java's behaviour |
|---|---|---|
| `read_level1_entry`'s `numImpactBytes` | `readShort` (signed) | `NegativeArraySizeException`/read failure |
| `read_full_block_header`'s impacts length | `readVInt` (decodes negative from a 5-byte varint) | same |
| `read_full_block_header`'s `blockLength` | `readVLong15` (signed `i64`) | seek out of range |

**Fixed**: `wire_length`/`add_wire_offset` (new, both returning
`Error::Store(Corrupted)`), and `read_length` — which `lucene-store` already
had, rejecting negatives *and* counts longer than the bytes remaining — for
the vint cases.

**Accumulators**: eight more sites, found by the Tier-2 review after this
batch's own audit stopped at the functions it was already editing — see the
review section for the list and for why leaving them was worse than it looks
(`read_positions` and the new walker decode the same bytes and are asserted
equivalent). All eight now `wrapping_add`, matching `decode_impacts_into`,
which already did.

Tests: `a_level1_entry_with_a_negative_impacts_length_is_rejected`,
`a_level0_header_with_a_negative_impacts_length_is_rejected`,
`a_level0_header_with_a_negative_block_length_is_rejected`. Each was run
against the un-fixed code to confirm it fails for the right reason: the
level-0 impacts case panics with `attempt to add with overflow` without the
guard. The accumulator sites have no dedicated test — an overflowing
accumulator needs a `.pos` block with `bitsPerValue == 32`, which no writer
here emits; they are a uniformity fix over a rule the file now states once.

### F3 `[CORRECTNESS → fixed]` Five allocations were sized by a value read off disk

The same audit, one class further out. An allocation whose size comes from the
file is worse than a panic: a failed allocation **aborts** the process, which
no `catch_unwind` at the FFI boundary can intercept.

- `read_postings_with_flags`: `Vec::with_capacity(doc_freq as usize)` ×2 —
  `docFreq` comes from the term dictionary.
- `decode_position_streams`: `Vec::with_capacity(total_term_freq as usize)` ×4
  — from the term dictionary, and negative values sign-extend.
- `decode_position_streams`' full-block payload run: `resize(start +
  read_vint())`.
- `decode_position_streams`' tail payload: `resize(start +
  last_payload_length)`.
- `read_positions`: `Vec::with_capacity(freq)` per document — `freq` from
  `.doc`.

**Fixed**: every reservation is now capped by something that actually exists
(the mapped file's own length, or the occurrences already decoded), and every
`resize` is bounded by the bytes remaining in the input (`read_length`, or an
explicit `remaining()` check). The reservation cap is a *hint* only, so no
real term pays anything: a document or a position never costs less than a byte
outside a packed block, and where a packed block really is denser the `Vec`
grows once more. Java never has this exposure because
`Lucene104PostingsReader` allocates fixed 256-entry buffers and streams.

Tests: `an_impossible_total_term_freq_is_rejected` (`-1` and `i64::MAX`,
through both the whole-term and the wanted-documents readers),
`a_tail_payload_longer_than_the_file_is_rejected`.

### F4 `[CORRECTNESS → fixed]` The lazy cursor's tail-block guard had no test

The main session fixed the panic (slicing the fixed 256-entry block array by a
`doc_count_left` that descends from `docFreq`) immediately before this batch;
`Lucene104PostingsReader.refillRemainder` asserts the same bound. It shipped
without a test, and the route to it is not obvious, so the guard could have
been deleted as dead defensive code by the next reader.

The route: a level-0 header whose `docDelta` overshoots what its body decodes
(nothing on the wire ties the two together). `advance` walks off the end of
the decoded block, `advance_shallow` returns early because the target is still
inside the block's *claimed* extent, and the tail path is entered with a whole
block's worth of documents still outstanding.

Test: `a_tail_block_larger_than_block_size_is_rejected_rather_than_panicking`.

### F5 `[CORRECTNESS → fixed]` `read_positions_for_docs` panicked on an out-of-range `wanted` entry

Its doc comment said "out-of-range indices are ignored, and duplicates simply
emit the same document twice". Neither was true: an index past the doc list
indexed the prefix-sum array out of bounds and panicked, and a duplicate
produced an *empty* second entry, not a repeat.

**Fixed**: `wanted_ranges` gives any entry the doc list does not have — and
any entry that is not strictly after the one before it, which covers both
duplicates and an unsorted `wanted` — an empty range that keeps its slot in
the result. The doc comment now says what the code does. Test:
`out_of_range_and_unsorted_wanted_entries_yield_empty_slots`.

### F6 `[PERF → fixed, measured]` A wanted-documents walk decoded every block it passed

**Java**: `advance(doc)` uses `.doc`'s skip data to jump, then `skipPositions`
decodes and discards only within the blocks it must.

**We did**: `read_positions_for_docs` ran `PForUtil` decode on *every* block of
`.pos` and `.pay` from the term's start to its end — 256-value bit-unpacks,
three of them per block for a field with offsets — keeping only the
occurrences it wanted, and never stopping early.

**Fixed**: a block holding no wanted occurrence is stepped over with
`PForUtil.skip` (one token byte and a seek per stream, which `for_util` already
had for `.doc`'s `DocsOnly` path), and the walk returns the moment the last
wanted document is behind it, leaving the rest of `.pos`/`.pay` unread.

### F7 `[PERF → fixed, measured]` The prefix-sum array was `docFreq + 1` entries whatever the caller asked for

Both wanted-documents readers built a `Vec<u32>` of every document's start
offset — 20 MB for a five-million-document term — to answer a question about
one document. Replaced by a single fused pass over `freqs` that emits the
ranges directly and keeps the same `freqs`-vs-`totalTermFreq` validation
(`wanted_ranges`). Worth 35.3 ms → 25.6 ms on the middle-document highlight
below.

### F8 `[PERF → fixed, measured]` `search_phrase_query` fetched every position of every term

The O15 defect class the brief asked about, one function over from the
highlighter. `search_phrase_query_scored_with_stats` already intersected the
doc lists first and asked only for the intersection's positions
(`positions_for_docs`); the **unscored** `search_phrase_query` still called
`term_doc_positions`, which materializes every position of every term through
`positions_flat` before the conjunction runs.

**Fixed**: the unscored path now has the scored path's shape, mechanically —
doc lists first (unfiltered, because `positions_for_docs` indexes the wire
stream by a running frequency sum that must total `totalTermFreq`; deletions
are applied to the candidate list), then one `positions_for_docs` per term for
the candidates. `span_doc_ids` deliberately keeps `positions_flat`: its
candidates are the *union* of its leaves, so the wanted set is the whole doc
list and there is nothing to save.

### F9 `[PERF]` Measurements

`benchmarks/.corpus/merged` (5.2 M documents, `body` indexed with offsets and
payloads), criterion, `cargo bench -p lucene-search --bench
highlight_offsets`. The A/B is in one build: `whole_term` is
`FieldTerms::positions` + index-of-doc, exactly what
`offsets_from_postings` called before; `one_doc` is
`FieldTerms::occurrences_for_doc`, what it calls now.

| term (`docFreq` / `totalTermFreq`) | document | `whole_term` (before) | `one_doc` (after) | |
|---|---|---|---|---|
| `t0` (4 997 130 / 56 600 329) | first | 1.3212 s | **18.344 ms** | 72.0x |
| | middle | 1.2952 s | **22.055 ms** | 58.7x |
| | last | 1.3040 s | **25.302 ms** | 51.5x |
| `t500` (4 711 / 4 713) | first | 168.29 µs | **5.6574 µs** | 29.7x |
| | middle | 163.18 µs | **5.7202 µs** | 28.5x |
| | last | 163.46 µs | **5.9631 µs** | 27.4x |
| `t999` (2 506 / 2 507) | first | 87.397 µs | **3.7345 µs** | 23.4x |
| | middle | 99.708 µs | **3.7221 µs** | 26.8x |
| | last | 87.537 µs | **4.1884 µs** | 20.9x |

Three positions per term because the new path's cost depends on where the
document sits (everything after it is never read) and the old path's does not
— which is exactly the asymmetry the fix introduces, and it shows: `t0`'s
first document costs 18.3 ms and its last 25.3 ms, while `whole_term` is flat
at ~1.30 s for all three.

The remaining `one_doc` cost is dominated by the term's **doc list**:
`postings()` on `t0` alone measures 5.69 ms (`docs_only_postings` bench,
re-run for this report), and the fused `wanted_ranges` pass over 5 M
frequencies is most of the rest. This port addresses `.pos` by a running
frequency sum and therefore needs every preceding document's frequency; Java
needs neither, because its skip data carries the `.pos` file pointers. That is
F10.

Phrase positions, same corpus and bench (`phrase_positions` group), for the
two-term phrase `t0 t1`: all-positions-then-intersect **627.16 ms** →
intersect-then-positions **493.99 ms**, 1.27x. Modest, and expected: `t0 ∩ t1`
is a large fraction of `t0`, so most blocks still hold a wanted document and
the doc-list decode dominates. The allocation saving is larger than the time
saving (the old shape materialized 56.6 M + 21 M positions, ~310 MB, per
query).

### F10 `[PERF — recorded, blocker named]` The `.pos`/`.pay` skip pointers are parsed and discarded

**Java**: `.doc`'s level-0 and level-1 skip data carries `posEndFPDelta` /
`posBufferUpto` (and the `.pay` pair), so `advance(doc)` seeks `.pos` straight
to the block containing the target document's occurrences. It never sums a
frequency and never reads a byte of the postings list it skipped.

**We do**: `read_full_block_header` and `read_level1_entry` parse those fields
for wire-order correctness and discard them (they say so). A wanted-documents
walk therefore starts at the term's `pos_start_fp` and steps block by block,
and — worse — needs the whole doc list to know *which* occurrence range a
document owns.

Cost after F6/F7: one token byte and a seek per 256 occurrences skipped, plus
the doc-list decode. Not fixed here because using the pointers means
accumulating them across every level-0 header from the term's start (i.e.
`advance_shallow`'s walk, on a second file) and threading a `.pos`/`.pay`
cursor position through `LazyDocsCursor` — a change to the shared block-header
reader that every other caller depends on, on top of a batch that already
rewrote this file's position walk. Recorded in `LEDGER.md`.

### F11 `[INTENTIONAL]` A batch walker where Java has a stateful cursor

Java's `PostingsEnum` is one cursor the caller advances. This port's callers
(phrase matching, the highlighter, span queries) all know their whole document
set up front, so the API takes a sorted `wanted` and returns a flat result
with per-document start indices. One pass, two allocations, no per-document
container — and the sink is a compile-time choice, so the positions-only walk
monomorphizes to exactly the code it had. Recorded rather than "fixed":
matching Java's shape here would cost more than it buys, and the wire decode
is identical either way.

### Verdict

Swept clean, with F10 open and recorded.

`postings.rs` is now at **97.85%** line coverage, above the repo's 95% bar. It
was **89.79%** at c8 — and **85.31%** after this batch's additions and before
its tests, because everything added lands in exactly the block c8 identified as
the gap: the `read_positions*` family, which only `lucene-search` reached, so a
two-crate coverage run never saw it. `tests/postings_wanted_docs.rs` reaches it
from `lucene-codecs` itself.

What was uncovered, and what still is (61 lines):

- **Was**: the whole `read_positions_flat` / `read_positions_for_docs` /
  `read_occurrences_for_docs` / walker block (~330 lines) — every wire path,
  every `wanted` shape, both sinks.
- **Still is**: `read_positions_flat`'s two frequency-sum guards (its wanted-
  documents siblings' equivalents are covered); the full-block payload-overrun
  guard and `add_wire_offset`'s error arm (both need a hand-built `.pay` whose
  lengths outrun its byte run); `read_tail_block`'s own
  `count >= BLOCK_SIZE` guard, which is now unreachable through any public
  entry point because `LazyDocsCursor::advance`'s guard catches it first (F4);
  the level-1 entry's `.pos`/`.pay` skip sub-fields, which need a
  `docFreq >= 8192` term on a positions-indexing field; `write_group_vints`'
  one-byte case; the test-only block-decode counter's `reset`; and
  `PostingsCursor`/`LazyDocsCursor` accessors only `lucene-search` calls
  (`level1_last_doc_id`, `current_block_last_doc_id`, `level0_last_doc_id`).

---

## `crates/lucene-index/src/check_index.rs`

Java: `CheckIndex.checkDocValueSkipper` (`CheckIndex.java:3667-3766`), called
from `checkDocValues` for every field whose
`docValuesSkipIndexType() != NONE`.

| Java check | Here |
|---|---|
| a fresh skipper reports `maxDocID(0) == -1` | `doc_values.skipper:<f>` (new) |
| `docCount() > 0 && minValue() > maxValue()` | `doc_values.skipper:<f>` (new) |
| `maxValueCount() < -1` | `doc_values.skipper:<f>` (new) |
| `docCount() == 0 && maxValueCount() != 0` | `doc_values.skipper:<f>` (new) |
| per interval: `minDocID(0) >= ` the doc advanced to | `doc_values.skipper:<f>` (new) |
| per level: `minDocID(level) <= maxDocID(level)` | `doc_values.skipper:<f>` (new) |
| per level: `minValue(level)`/`maxValue(level)` nested in the global range | `doc_values.skipper:<f>` (new) |
| per level: `minValue(level) <= maxValue(level)` | `doc_values.skipper:<f>` (new) |
| `sum(docCount(0)) == docCount()` | `doc_values.skipper:<f>` (new) |

### F12 `[MISSING → fixed]` `checkDocValueSkipper` was unported

c9 finding 11, recorded rather than rushed because the skipper's read API was
`advance`-shaped and did not exist yet. b6 has since ported
`DocValuesSkipper` with Java's exact `advance` state machine, so the structures
were all there.

Why it matters more than most missing checks: a skipper is a *promise about
documents the reader will then not look at*. `DocValuesSkipper.advance` trusts
every level's `[minDocID, maxDocID]` and `[minValue, maxValue]` to skip whole
subtrees, so a skipper whose bounds are too narrow silently drops matching
documents from every range query that uses it — and nothing else in
`CheckIndex` notices, because the per-document values all still decode
correctly out of `.dvd`. The `.dvs` file was opened and fully CRC-verified
(c9 finding 16) and never interpreted.

**Fixed**: `check_doc_value_skipper`, run from `check_doc_values` for exactly
the fields Java runs it for, over the `.dvs` opened once alongside `.dvm`/
`.dvd`. The interval walk carries an iteration bound of `intervals.len() + 1`
— every iteration consumes at least one interval, so it is a bound and not a
heuristic — because `check_index` must not be able to hang on a corrupt file.

Tests: `skipper_checks_actually_run_on_the_skip_index_fixture`, and the
negative control below.

### F13 `[CORRECTNESS of the test suite]` The negative control had to re-sign the footer

Every other negative control in this module flips a byte and leaves the footer
stale — which the `file:*` checksum check catches on its own, so it proves
nothing about the semantic check under test. For a `.dvs` that is worse than
useless: `parse_skip_index` validates the footer itself, so *every* flip would
have been "caught" by the new check without any of its invariants firing.

`corrupting_the_doc_values_skipper_is_caught_by_the_skipper_check` recomputes
the footer over the corrupted bytes, so the file is perfectly well-formed and
only the skip index's own invariants can catch it. Result over the whole
`.dvs` body of `fixtures/data/doc_values_skip_index` (2 masks per byte):
**428 of 574 corruptions rejected**, 146 semantically harmless (a value bound
moved but stayed nested inside the global range). All 574 were accepted before
this batch. One of the rejected ones is then put through the whole of
`check_directory`, which must report `doc_values.skipper:<f>` **and nothing
else** — proving both the wiring and that the corruption really is invisible
to every other check.

The sweep itself runs against `check_doc_value_skipper` directly rather than
`check_directory`, which turned a 501-second test into a fast one.

### F14 `[MISSING → fixed]` `.nvm` entries were never validated against `FieldInfos`

See `norms.rs` below for the decision; the check itself is
`norms.entries_name_real_norms_fields`, in this file, which is where
`CheckIndex` puts it in Java too. Negative control:
`a_nvm_entry_for_a_nonexistent_field_is_caught` (footer re-signed, same
discipline as F13).

### F15 `[INTENTIONAL — mislabel corrected]` The `hnsw.*` checks are not a port of 10.5.0

Handed over by `c18-version-audit`, and confirmed here against the pinned
tree: `/home/tuong/work/lucene-10.5.0/.../CheckIndex.java` contains **no**
`testHnswGraphs` (`grep -c` = 0); `main` grew it later. c9 documented the
`hnsw.*` block as a port of it, in the module doc, in `check_vectors`' doc and
in `check_hnsw_graphs`' doc.

The checks stay: they are diagnostic-only, they catch real corruption
(`corrupted_hnsw_graph_bytes_are_never_silently_accepted`,
`corrupted_hnsw_neighbours_are_caught_by_the_graph_checks`), and the three
properties they check are the ones Lucene's own later `testHnswGraph` settled
on. Only the label was wrong, and all three doc sites now say so. `testVectors`
*is* a 10.5.0 check and remains correctly labelled.

### F16 `[INTENTIONAL — mislabel corrected]` The `Float16` scope note was a `main`-ism

Also from `c18`: the module's "still out of scope" list named the `Float16`
vector encoding. 10.5.0's `VectorEncoding` has exactly `BYTE` and `FLOAT32`, so
there is no third encoding for this module to be out of scope about. Note
rewritten to say that, with the fact that this port's reader rejects a third
ordinal. (c18 had already fixed the `lucene-codecs` copy; this was the
`check_index.rs` one.)

### Verdict

Swept clean for this batch's scope. c9's remaining recorded items
(`estimatePointCount`, `checkDVIterator`) are unchanged and still
`INTENTIONAL`.

**Coverage note, not this batch's**: `check_index.rs` is at 89.77% line
coverage, below the repo's 95% bar. It was **89.59% before this batch** and the
243 lines added here are 93% covered, so this is c9's shortfall inherited, not
one introduced — but it is a second file under the bar (alongside
`checksum_verify.rs` at 93.75%, which AGENTS.md already names) and it should
be recorded as such rather than discovered again.

---

## `crates/lucene-codecs/src/norms.rs`

Java: `Lucene90NormsProducer.readFields` (`Lucene90NormsProducer.java:78-110`).

| Rust | Java | Verdict |
|---|---|---|
| `parse_meta` | ctor + `readFields` (the byte decode) | identical, incl. the `0/1/2/4/8` `bytesPerNorm` rejection |
| `validate_fields` | `readFields`' `FieldInfos` checks | **added** (F14) |

### F14 (cont.) `[MISSING - fixed]` — and why `parse_meta`'s signature did not change

Raised by b6 (#4), deferred by c7 (F-23) with reasoning, and the brief asked
for a decision rather than a third deferral.

**Java**: `readFields(IndexInput meta, FieldInfos infos)` throws
`CorruptIndexException` for an entry naming a field number the `.fnm` does not
have, and for one naming a field whose `FieldInfo.hasNorms()` is false
(`indexOptions == NONE || omitNorms`).

**We did**: `parse_meta` takes no `FieldInfos`, so both are accepted. The
consequence is silent rather than loud: the entry becomes unreachable, so
every norm lookup for the *real* field falls back to "this field has no norms"
and every score for it is computed against a default norm.

**Decision — the diagnostic is delivered; the signature change is declined.**
`parse_meta` has **23 call sites** outside its own module, across four crates:
`lucene-codecs/tests/norms_fixtures.rs` (6), `lucene-search` (src 2, tests 3,
benches 1), `lucene-index` (`check_index.rs` 1, `merge.rs` 5,
`index_writer.rs` 2), `lucene-ffi` (2). Three of those files are owned by
in-flight batches (c10's `index_writer.rs`, c14's `merge.rs`, c13's
`lucene-ffi`), and more than half of the sites are hand-built round-trip tests
with no `FieldInfos` to pass and nothing to gain from one. Threading a
parameter through all 23 to reach the two that matter is the wrong trade —
c7's assessment was right about the cost.

What was wrong with c7's conclusion is that it treated the signature and the
diagnostic as the same decision. They are not. `norms::validate_fields(&Norms,
&FieldInfos)` is additive, and the two call sites where Java's diagnostic
actually fires now call it:

- `directory_reader.rs`' segment open — the same moment, the same path and the
  same consequence as Java's (the segment fails to open), one line after
  `parse_meta`;
- `check_index.rs`' norms checks — which is `CheckIndex`'s own job.

Aligning `parse_meta`'s signature with Java's remains a worthwhile tidy-up
when a shared `FieldInfos` open exists, but it is now a refactor with no
behaviour attached, not a missing check. Closed, not deferred.

Tests: `validate_fields_tests::{entries_naming_real_norms_fields_are_accepted,
an_entry_for_a_field_the_fnm_does_not_have_is_rejected,
an_entry_for_a_field_with_no_norms_is_rejected}` (both halves of
`hasNorms()`), plus `check_index`'s `a_nvm_entry_for_a_nonexistent_field_is_caught`.

### Verdict

Swept clean. b6 #4 / c7 F-23 closed.

---

## Tier-2 review (`quality-reviewer`)

Run on this batch's scope (the reviewer read the Java alongside, and
independently re-derived the wanted-documents walk against
`refillPositions`/`refillOffsetsOrPayloads`/`refillLastPositionBlock`/
`nextPosition` and `PForUtil.skip` rather than against this batch's comments).
It confirmed the decode equivalence, the per-document accumulator resets, the
per-block payload byte-run arithmetic, every `wanted` shape, the early exit,
`checkDocValueSkipper` check-for-check, `validate_fields` against
`hasNorms()`, and the two `lucene-search` migrations.

**One gating finding, and it was the important one**: the F2/F3 audit stopped
at the sites this batch was already touching and **missed eight of the same
class** — accumulators, not lengths. All eight are now `wrapping_add`, with
the reason stated once and referenced from the rest:

- `read_positions`' `position`, `start_offset_acc` and its `end` (`i32 +` a
  `.pos` value widened from `u32`), and `read_positions_flat`'s `position`;
- `read_full_block_header`'s `prev_doc_id + doc_delta` (a `readVInt15` off
  disk), in the same function whose *lengths* this batch had just hardened;
- both `level1_last_doc_id += entry.doc_delta` sites;
- `decode_full_block_body`'s consecutive-docs path (`prev_doc_id + 1 + i`);
- `LazyDocsCursor::skip_level1_to`'s `doc_freq - level1_doc_count_upto` /
  `+= LEVEL1_NUM_DOCS`, and `decode_term_metadata`'s singleton
  `prev.singleton_doc_id as i64 + delta`.

The reviewer's sharpest point is worth keeping: `read_positions` and the new
walker decode the same bytes, and `postings_wanted_docs.rs` asserts they agree
— so leaving one wrapping and the other panicking makes them *disagree
precisely on the corrupt input the batch was hardening against*. The rule is
now uniform across the file, and `decode_impacts_into` (which already wrapped)
was the precedent.

Its suggested mechanical guard is recorded as a carry-over: nothing catches
this class today, because a release build is silent and the debug panic needs a
corrupt fixture to fire.

Five advisories, all acted on: a doc comment this batch's insertion had spliced
onto `corrupted` (three paragraphs about `decode_full_block_body` and
`check_wire_position`, the latter left with no doc at all) — unspliced; a
garbled 22-space corruption message inherited from c8 — fixed; `wire_count`
reporting a legal-but-unsupported `totalTermFreq > u32::MAX` as *corruption*
when it is this port's own ceiling (`totalTermFreq` is a `long` in Lucene, and
a stop-word in a 200 M-document segment really can exceed 2^32 occurrences) —
now `Error::Unsupported` with the limit named, and the test asserts the two
kinds separately; `occurrences_for_doc`'s "an empty `Vec` cannot occur"
overclaiming (a `.doc` claiming `freq == 0` produces one) — reworded; and the
`.dvs` negative control asserting `caught > 0` where it measures 428 — now a
floor of 400 with the measured number in the message, so the check's *reach* is
pinned and not just its existence. The same `> 0` weakness in c9's
`corrupted_sorted_dv_ordinal_space_is_caught` is left alone and recorded: it is
in this file but not this batch's check, and setting its floor honestly means
measuring it first.

## Cross-file notes

- **`blocktree.rs`** (unowned, additive only): `FieldTerms::occurrences_for_docs`
  and `FieldTerms::occurrences_for_doc`. No existing method changed.
- **`highlighter.rs`** (`lucene-search`, unowned): `offsets_from_postings`
  migrated to `occurrences_for_doc`; the `PostingsFlags::DocsOnly` doc-list
  lookup c12 added is gone because the new path needs the frequencies it
  deliberately skipped — and gets them from the one `postings()` call it now
  makes instead of two.
- **`lib.rs`** (`lucene-search`, unowned): `search_phrase_query` migrated to
  the scored path's docs-first shape (F8).
- **`directory_reader.rs`** (`lucene-search`, unowned): one added
  `norms::validate_fields` call (F14).

## Gates

- `cargo fmt --all` — clean.
- `cargo clippy -p lucene-codecs -p lucene-index -p lucene-search --all-targets -- -D warnings` — clean.
- `cargo test -p lucene-codecs -p lucene-index` — **1 850 passed, 0 failed**.
- `cargo test -p lucene-search` — **1 013 passed, 0 failed** (this batch edits
  three of its files and one of its benches).
- `python3 scripts/check-parity.py` — ok (the new gate from c18's handoff).
- `scripts/verify-write-path.sh` — **20/20**, confirmed rather than assumed. It
  was 18/18 at c8; `c14` added `VerifyDocValuesUpdates` and `c10`
  `VerifySortedSegment` while this batch ran.
- `cargo llvm-cov -p lucene-codecs -p lucene-index` — `postings.rs` **97.85%**,
  `norms.rs` **98.67%**, `blocktree.rs` **96.94%**, two-crate total **97.25%**.
  `check_index.rs` is **89.75%**, below the bar and inherited (see its
  verdict).

## Carry-over items raised by this batch

- [ ] **Nothing mechanical catches the "arithmetic on a file value" class.**
      Five defects in `postings.rs` alone (b5 F2, c8's four, the main session's
      tail block), and this batch's own audit still missed eight sites until
      the Tier-2 review found them. A release build is silent and the debug
      panic needs a corrupt fixture, so review is the only thing standing
      between this class and the JVM. `clippy::arithmetic_side_effects`, denied
      at module level with `#[allow]` on the arithmetic that is provably ours,
      would catch all of it — suggested by the reviewer, worth an experiment on
      `postings.rs` first before considering it workspace-wide.
- [ ] **`check_index.rs` is at 89.77% line coverage**, below the repo's 95%
      bar (it was 89.59% before this batch). A second file under the bar
      alongside `checksum_verify.rs` at 93.75%, which AGENTS.md already names;
      this one is not named anywhere.
- [ ] **`corrupted_sorted_dv_ordinal_space_is_caught` asserts
      `caught_by_ords > 0`**, which survives a regression to a single caught
      byte. Same weakness this batch fixed in its own `.dvs` control; fixing
      c9's honestly means measuring its real count first.

- [ ] **The `.pos`/`.pay` skip pointers in `.doc`'s level-0/level-1 records are
      still parsed and discarded** (F10), so a wanted-documents walk still
      needs the term's whole doc list and still steps block by block through
      `.pos`. Blocker: threading a `.pos`/`.pay` cursor through the shared
      block-header reader and `LazyDocsCursor`.
- [ ] **`norms::parse_meta`'s signature still differs from Java's**
      (`readFields(meta, infos)`), now a pure tidy-up with no behaviour
      attached (F14). Do it when a shared `FieldInfos` open lands.
