# c14-dv-updates-format

Follow-up batch opened from one carry-over: c7's **F-15**, "a segment carrying
doc-values updates is not real-Lucene-readable (the overlay format is this
port's own — b6's declared scope)". That is the same defect shape the sweep has
already closed twice — b11's invented `.si` index-sort encoding and c5's
invented `.vec`/`.vem` layout — and it is a correctness-of-contract failure, not
a performance one: this port exists to be read by OpenSearch's Lucene.

**Java source of truth.** `/home/tuong/work/lucene` at tag
`releases/lucene/10.5.0`, **not** its working tree. This matters here more than
anywhere else in the sweep so far: `main` carries a *different, newer*
`ReadersAndUpdates` that writes doc-values updates as **overlay deltas** with a
fold-to-dense compaction ratio. Porting from the working tree would have
produced a format 10.5.0 cannot read — the very failure this batch exists to
fix. Every Java line quoted below is from the 10.5.0 tag.

| Rust file | Java counterpart(s) |
|---|---|
| `crates/lucene-index/src/field_updates.rs` (**new**) | `index/ReadersAndUpdates.java` (`writeFieldUpdates`, `handleDVUpdates`, `writeFieldInfosGen`, `cloneFieldInfo`), `index/IndexFileNames.java` (`fileNameFromGeneration`, `segmentFileName`), `codecs/perfield/PerFieldDocValuesFormat.java` (`getSuffix`, `getFullSegmentSuffix`, `FieldsWriter.getInstance`) |
| `crates/lucene-codecs/src/doc_values_updates.rs` | `index/DocValuesFieldUpdates.java` (`mergedIterator`), `index/{Numeric,Binary}DocValuesFieldUpdates.java`, `ReadersAndUpdates.MergedDocValues` |
| `crates/lucene-codecs/src/doc_values.rs` (write side) | `codecs/lucene90/Lucene90DocValuesConsumer.java` (`writeValues`' and `addBinaryField`'s `numDocsWithValue == 0` branch) |
| `crates/lucene-index/src/segment_infos.rs` | `index/SegmentCommitInfo.java` (`advanceFieldInfosGen`, `advanceNextWriteFieldInfosGen`, `advanceNextWriteDocValuesGen`, `setDocValuesUpdatesFiles`) |
| `crates/lucene-index/src/index_writer.rs` (two small edits) | `ReadersAndUpdates.writeFieldUpdates`' call site; `MergePolicy` source selection |
| `crates/lucene-search/src/directory_reader.rs` (one small edit) | `index/SegmentDocValuesProducer.java`, `index/SegmentDocValues.java`, `SegmentReader.initFieldInfos`/`initDocValuesProducer`, `IndexWriter.readFieldInfos` |

---

## What Lucene 10.5.0 actually does

Worth stating up front, because the invented format was a *plausible*
misunderstanding of it and the whole batch is a correction of that
misunderstanding.

A doc-values update is **not a delta file**. `ReadersAndUpdates.handleDVUpdates`
rewrites the updated field's **whole column** — the reader's current values
merge-sorted with the resolved updates (`MergedDocValues` over
`DocValuesFieldUpdates.mergedIterator`) — through
`Lucene90DocValuesConsumer`, into a brand-new generation of *ordinary*
doc-values files. The base `.dvm`/`.dvd`/`.dvs` are left untouched but
superseded for that field, and are still read for every field the update did not
touch:

```text
_0.si                          the segment, unchanged
_0_Lucene90_0.dvm/.dvd/.dvs    the base column: still there, still read for
                               every field this update did not touch
_0_3_Lucene90_0.dvm/.dvd/.dvs  generation 3 of `val`'s column
_0_2_Lucene90_0.dvm/.dvd/.dvs  generation 2 of `tag`'s column
_0_3.fnm                       FieldInfos generation 3: val.docValuesGen=3,
                               tag.docValuesGen=2, keep.docValuesGen=-1
```

Four things have to agree for a reader to find any of it, and **every way of
getting one of them wrong reads back plausibly through this port's own reader**:

1. the segment suffix, `PerFieldDocValuesFormat.getFullSegmentSuffix(
   Long.toString(gen, 36), "Lucene90_0")` — used both in the file *name* and
   inside each file's *index header*;
2. `FieldInfo.docValuesGen` in a new `FieldInfos` generation
   (`writeFieldInfosGen`), which is the only thing that tells a reader a
   generation exists;
3. `SegmentCommitInfo.docValuesGen`/`fieldInfosGen`;
4. `SegmentCommitInfo.getDocValuesUpdatesFiles()`, keyed by field number, which
   is what makes the deleter, `CheckIndex` and `checksum_verify` see the files.

That is the shape this batch ports, and it is exactly what the fixture
generator's output confirms Lucene 10.5.0 writes (see the last section).

---

## `crates/lucene-index/src/field_updates.rs` (new)

### Method correspondence

| Rust | Java | Verdict |
|---|---|---|
| `write_field_updates` | `ReadersAndUpdates.writeFieldUpdates(Directory, FieldNumbers, long maxDelGen, InfoStream)` | ported; the `maxDelGen` filtering and the pending-packet pruning stay in `index_writer.rs`, which is where this port's packets live (F-11) |
| `write_field_updates_inner` | the same method's `try` body + `handleDVUpdates` | ported, per-field loop and all |
| `finish_generation` | `handleDVUpdates`' per-field tail (`info.advanceDocValuesGen()` + `fieldFiles.put`) + `PerFieldDocValuesFormat.getInstance`'s `putAttribute` pair | identical |
| the `Err` arm of `write_field_updates` | `finally { if (success == false) { advanceNextWriteFieldInfosGen(); advanceNextWriteDocValuesGen(); deleteFilesIgnoringExceptions(...) } }` | identical (F-5, F-13): Java has no snapshot to restore, so the counters are carried across this port's restore at whichever value is higher |
| `read_current_field_infos` | `IndexWriter.readFieldInfos(SegmentCommitInfo)` + `ReadersAndUpdates`' `cloneFieldInfo` loop over `reader.getFieldInfos()` | identical outcome |
| `read_current_column` / `read_base_numeric` / `read_base_binary` | `reader.getNumericDocValues(field)`/`getBinaryDocValues(field)` resolved through `SegmentDocValuesProducer` | identical resolution rule (`FieldInfo.docValuesGen`, one-field `FieldInfos` per generation) |
| `generation_file_name` / `generation_segment_suffix` / `field_infos_gen_file_name` | `IndexFileNames.fileNameFromGeneration` + `PerFieldDocValuesFormat.getFullSegmentSuffix`/`getSuffix` | identical (F-1) |
| `check_updatable` | `handleDVUpdates`' `assert type == NUMERIC || type == BINARY` + `assert fi.getDocValuesType() == update.type` | ported as errors rather than assertions (F-11) |
| `field_per_field_component` / `per_field_component` | `PerFieldDocValuesFormat.getInstance`'s `PER_FIELD_FORMAT_KEY`/`PER_FIELD_SUFFIX_KEY` lookup | identical rule (F-12) |
| `format_name` / `suffix_component` / `put_attribute` / `base_codec_suffix` | `PerFieldDocValuesFormat.getSuffix`'s inverse; `FieldInfo.putAttribute` | not-in-Java helpers |
| — | `pendingDVUpdates`/`mergingDVUpdates`, `isMerging`, `sortMap`, `swapNewReaderWithLatestLiveDocs`, `ramBytesUsed`, `TrackingDirectoryWrapper`, `IOContext.flush(FlushInfo)`, `InfoStream` messages | **not ported**: the reader pool and RAM accounting c7 F-14/F-6 already recorded as absent, plus Java-shape plumbing |
| — | `FieldInfos.FieldNumbers.constructFieldInfo` (create a field the segment never had) | not needed: this port's field list is fixed at `IndexWriter::open`, so the `FieldInfo` already exists — c7 F-13's recorded divergence. The *column* still legitimately does not exist, and that case is ported and tested |

### Findings

#### F-1 `[CORRECTNESS]` — the on-disk format was this port's own invention (c7 F-15). **Fixed.**

**Java**: as described above — a generation-suffixed `.dvm`/`.dvd`/`.dvs`
triple holding the field's whole rewritten column, named
`_<segment>_<base36 gen>_Lucene90_0.<ext>` and carrying that same
`"<base36 gen>_Lucene90_0"` string in each file's index header.

**We did**: one file per `(generation, field)` in `doc_values_updates.rs`'s own
encoding — a sparse `(docId, hasValue, value)` delta list — named
`<segment>_<base36 gen>_LuceneRustDVU_<fieldNumber>.dvu`, recorded in
`dv_update_files`.

**Consequence**: an index *with* a doc-values update could not be handed to real
Lucene at all. `CheckIndex` and any doc-values read of that field would open the
recorded files as `Lucene90DocValuesFormat` output and fail. Since OpenSearch's
soft-delete retention, `_seq_no` and `_primary_term` all ride on doc-values
updates, this was not an exotic corner: it is the normal steady state of an
OpenSearch shard.

**Fixed**: `field_updates::write_field_updates`. `merge_numeric_column`/
`merge_binary_column` build the merged column, `write_numeric_generation`/
`write_binary_generation` write it through the existing (already
Lucene-verified) `doc_values.rs` writers with the generation suffix, and the
naming helpers reproduce `fileNameFromGeneration` + `getFullSegmentSuffix`.

**Proof, in both directions** — a self-round-trip proves nothing here, which is
exactly how the invented format survived:

- **Rust writes, Lucene reads**: `crates/lucene-index/examples/write_doc_values_updates_fixture.rs`
  + `fixtures/src/VerifyDocValuesUpdates.java`, wired into
  `scripts/verify-write-path.sh` (**18/18 → 19/19**; the 18 baseline was
  confirmed by running it before touching anything). Real Lucene opens two
  Rust-written indices through `DirectoryReader`, reads **every** document's
  updated value, checks `SegmentCommitInfo`'s recorded generations against the
  directory, and runs `CheckIndex` at `MIN_LEVEL_FOR_SLOW_CHECKS`.
- **Lucene writes, Rust reads**: `fixtures/src/GenDocValuesUpdates.java` +
  `crates/lucene-index/tests/doc_values_updates_fixtures.rs` and
  `crates/lucene-search/tests/doc_values_updates_reader_fixtures.rs`.

**Verified to discriminate.** With `merge_numeric_column` altered to start from
an all-absent column instead of the base (the single most likely way to get a
full-column rewrite wrong), the verifier fails with
`MISMATCH numeric column density: 130 documents have a value, expected 260`.
Nothing structural catches that — the files are well-formed and checksummed;
only reading the values back does.

#### F-2 `[MISSING]` — no `FieldInfos` generation, so nothing could point at a generation. **Fixed.**

**Java**: `writeFieldUpdates` ends with
`fieldInfosFiles = writeFieldInfosGen(fieldInfos, trackingDir, codec.fieldInfosFormat())`,
which writes `_<segment>_<base36 fieldInfosGen>.fnm` (suffix in the header too),
calls `info.advanceFieldInfosGen()`, and records the file set on the
`SegmentCommitInfo`. `IndexWriter.readFieldInfos` and
`SegmentReader.initFieldInfos` then read *that* file, because
`FieldInfo.docValuesGen` is the only place a reader learns a generation exists.

**We did**: nothing. `SegmentCommitInfo.field_infos_gen`/`field_infos_files`
existed (c7 F-18 added the fields) and were never written by anything.

**Fixed**: `write_field_updates_inner`'s tail, plus
`SegmentCommitInfo::advance_field_infos_gen`. The generational `.fnm` carries
the full field list with the updated fields' `doc_values_gen` stamped and, for a
field the base flush wrote no column for, the `PerFieldDocValuesFormat.format`/
`.suffix` attributes added (`PerFieldDocValuesFormat.getInstance`'s
`putAttribute` pair) — without those, no reader registers a producer for the
field at all and the generation is on disk, referenced, checksummed, and never
read.

Tests: `a_field_with_no_base_column_gains_the_per_field_format_attributes`,
`the_generational_field_infos_records_each_fields_own_doc_values_generation`
(against real Lucene's own `.fnm`).

#### F-3 `[CORRECTNESS]` — `dvUpdatesFiles` accumulated where Java replaces. **Fixed.**

**Java**: `writeFieldUpdates` copies forward only the fields *not* updated in
this session, then `info.setDocValuesUpdatesFiles(newDVFiles)` — so a field's
entry is **replaced** by the new generation's files.

**We did**: `record_dv_update_file` **pushed** onto the field's list, with a doc
comment asserting the opposite of Java ("an older generation's overlay stays
referenced, because a reader still has to apply it underneath the newer one").
That was true of the *delta* format and is false of the real one.

**Consequence**: every superseded generation would stay referenced by
`segments_N` forever. `IndexFileDeleter` would never reclaim it (it is
referenced), `checksum_verify` and `CheckIndex` would keep verifying it, and a
long-lived shard would accumulate one dead column per update round — unbounded
growth from a bookkeeping mistake, with no error anywhere.

**Fixed**: `SegmentCommitInfo::set_doc_values_updates_files` replaces.

**Test**: `successive_doc_values_updates_supersede_each_other_at_a_new_generation`
asserts the recorded files are generation 2's and that no `_0_1_*` file remains
in the directory at all; `VerifyDocValuesUpdates.verifyGenerationBookkeeping`
asserts the same thing from Java's side over three rounds. (This is also visible
in the Java fixture itself: `GenDocValuesUpdates` updates `val` twice and
generation 1's files are simply gone.)

#### F-4 `[CORRECTNESS]` — one `docValuesGen` per update round, where Java takes one per field. **Fixed.**

**Java**: `info.advanceDocValuesGen()` is **inside** `handleDVUpdates`'
`for (Entry<String, List<DocValuesFieldUpdates>> ent : pendingDVUpdates)` loop.
Two fields updated in one round therefore land at two different generations.
They must: each generation's `.dvm` describes exactly *one* field, and
`SegmentDocValuesProducer` asserts `!dvGens.contains(docValuesGen)` precisely
because two fields sharing a generation would mean one `.dvm` claiming to be
two different single-field files.

**We did**: `advance_doc_values_gen()` once, then wrote every field's file under
that one generation.

**Consequence**: with two fields updated in one round the two `.dvd`s collide on
the same file name — the second write silently overwrites the first, and one
field's column is simply the other's.

**Fixed**: the generation is taken and advanced inside `finish_generation`, per
field.

**Test**: `each_updated_field_takes_its_own_doc_values_generation` (two fields
in one round → `doc_values_gen == 2`, `field_infos_gen == 1`, and the
generational `.fnm` records 1 and 2 for the two fields). The Java fixture
independently confirms the rule: `val` is at generation 3 and `tag` at
generation 2.

#### F-5 `[MISSING]` — no failure path; a half-written round left `SegmentCommitInfo` naming files that do not exist. **Fixed.**

**Java**: `finally { if (success == false) { info.advanceNextWriteFieldInfosGen();
info.advanceNextWriteDocValuesGen(); for (String fileName :
trackingDir.getCreatedFiles()) IOUtils.deleteFilesIgnoringExceptions(dir,
fileName); } }`. Two things: the partial files go, and both *next-write*
counters step so a retry cannot reuse a name the failed attempt may already have
created.

**We did**: nothing — c7's writer advanced the generation and wrote files with
no unwind at all, so an error part-way through a multi-field round left the
in-memory `SegmentCommitInfo` referencing files the next commit would name and
the deleter would not find.

**Fixed**: `write_field_updates` snapshots the `SegmentCommitInfo`, and on error
restores it, advances both next-write counters
(`SegmentCommitInfo::advance_next_write_field_infos_gen`/
`advance_next_write_doc_values_gen`, both newly ported) and deletes every file
the attempt created.

**Test**: `a_failure_part_way_through_rolls_the_whole_round_back` — a round whose
*second* field fails after the first has already been written; asserts the
generations are back at `-1`, `dv_update_files`/`field_infos_files` are empty,
the partially written `.dvd` is gone from disk, and the retry lands at
generation **2**, not back at 1.

#### F-6 `[MISSING]` — the read path never resolved an update generation at all. **Fixed.**

**Java**: `SegmentReader.initDocValuesProducer` builds a
`SegmentDocValuesProducer` when `si.hasFieldUpdates()`, which maps **each
field** to the producer for *its own* `FieldInfo.docValuesGen` — the base
producer (gen `-1`) for fields that were never updated, and a one-field producer
per generation for the fields that were. `SegmentReader.initFieldInfos` reads
the generational `.fnm` first, because that is where those generations are
recorded.

**We did**: `SegmentReader::open` read the base `.fnm` unconditionally and the
base `.dvm`/`.dvd` unconditionally. Nothing anywhere resolved a doc-values
update generation — including the `.dvu` overlays c7's writer produced, which
had **no reader** outside unit tests. A committed doc-values update was
invisible to this port's own reader.

**Fixed**: `SegmentReader::open` reads the generational `.fnm` when
`commit.field_infos_gen != -1`, then opens one `(meta, data)` pair per field
whose `doc_values_gen != -1` (deriving the per-field codec component from the
field's own `.fnm` attributes, as `PerFieldDocValuesFormat.FieldsReader` does,
and skipping a field with no format attribute exactly as Java does). The new
`SegmentReader::doc_values_for_field(field_number)` is
`SegmentDocValuesProducer.dvProducersByField`; `doc_values_meta`/
`doc_values_data` stay the *base* pair so no existing caller changes behaviour
for a field with no generation.

b13's reader-reuse predicate already keyed on `field_infos_gen`/
`doc_values_gen`, so a reopen after an update correctly re-opens rather than
serving the stale reader — that half was already right and is unchanged.

**Test**: `crates/lucene-search/tests/doc_values_updates_reader_fixtures.rs`,
over real Lucene's own three-round-updated index: `val` (updated twice) and
`tag` (updated once, *different* generation) read back their newest values,
`keep` (never updated) reads back from the base column, and the base column is
separately confirmed to still hold `val`'s pre-update value — so "resolves to
the generation" is distinguished from "happens to agree".

#### F-7 `[MISSING]` — `Lucene90DocValuesConsumer`'s "no documents with values" shape was not writable. **Fixed.**

**Java**: `writeValues` and `addBinaryField` both branch on
`numDocsWithValue == 0` and write `meta[-2, 0]` — `docsWithFieldOffset = -2`,
length 0, `jumpTableEntryCount = -1`, `denseRankPower = -1` — instead of an
`IndexedDISI` structure. `addBinaryField` additionally leaves `minLength` at
`Integer.MAX_VALUE` and `maxLength` at 0, which is what makes
`maxLength > minLength` false so no address array follows.

**We did**: `write_single_sparse_numeric_field`/`write_single_sparse_binary_field`
always wrote an `IndexedDISI` structure, and the binary one wrote
`min_length = 0` for an empty value set. The port's *reader* already handled
`-2` (`NumericEntry::is_empty_field`), so this was a write-side-only gap.

**Reachable from this batch**: `update_doc_values` with a `None` value against a
term matching every document resets the whole column, so the merged column has
zero present docs. Before the fix that produced a zero-block DISI region that a
reader would rank-index into an empty value array rather than short-circuiting.

**Fixed**: `write_empty_docs_with_field` in `doc_values.rs`, taken by both
sparse writers when there are no present docs, with Java's exact `minLength`/
`maxLength` for the binary case.

**Tests**: `a_sparse_numeric_field_with_no_values_writes_javas_empty_marker`,
`a_sparse_binary_field_with_no_values_writes_javas_empty_marker` (both assert
the four marker values *and* that every doc reads back as absent), plus
`a_generation_that_reset_every_value_round_trips_as_an_empty_column` and
`a_reset_of_every_value_yields_an_empty_column_not_a_column_of_zeroes` from the
two layers above.

#### F-8 `[CORRECTNESS]` — a segment carrying only an update generation was auto-mergeable, and the merge would have dropped it. **Fixed.**

**Java**: not applicable in this shape — Java's merge is doc-values-aware
(`ReadersAndUpdates` carries `mergingDVUpdates` forward onto the merged
segment). This port's `execute_merge` has no doc-values path at all, which
`segment_stats` already knew: it excludes any segment whose `.si` lists a
`.dvd`, with a doc comment explaining that merging one would silently drop the
column.

**We did**: test only the `.si`'s file list. A doc-values *update* against a
field the base flush wrote no column for lives **entirely** in generational
files, and no `.si` lists those — they did not exist when it was written. Such a
segment passed the guard.

**Consequence**: an automatic merge would consume it and produce a merged
segment with no doc-values at all. Exactly the silent data loss the existing
guard was written to prevent, reached through the one door it did not cover.
(Pre-existing: the same hole admitted c7's `.dvu` files.)

**Fixed**: `segment_stats` also skips a segment with `doc_values_gen != -1`.

**Test**: `a_segment_carrying_only_a_doc_values_update_generation_is_never_auto_merged`
— asserts up front that the `.si` genuinely lists no `.dvd` (so the old guard
would have admitted it), then runs six commits under a tight merge policy and
requires the updated segment to survive with its value intact. **Verified to
discriminate**: with the new guard removed it fails at "the updated segment must
survive every merge round".

#### F-9 `[PERF]` — the merged column is materialised, not merge-sorted

**Java**: `MergedDocValues` is a two-way merge of the reader's
`NumericDocValues` iterator and `DocValuesFieldUpdates.mergedIterator(subs)` (a
priority queue over the packets), streamed straight into the consumer. Nothing
is materialised: peak extra memory is the packets themselves.

**We do**: `merge_numeric_column`/`merge_binary_column` build a
`Vec<Option<i64>>` / `Vec<Option<Vec<u8>>>` of length `maxDoc` and index into
it.

**Cost, stated rather than hand-waved**: `8 + 8` bytes/doc for numeric
(`Option<i64>` is 16 bytes), so ~16 MB for a 1M-doc segment, transient for the
duration of one field's rewrite; binary additionally clones each value once.
Java pays neither. Against that: the *dominant* cost of a generation is the
`Lucene90DocValuesConsumer` pass itself, which is `O(maxDoc)` in both, and this
port's `doc_values.rs` writers take slices, not iterators — feeding them a
merge-sorted iterator would mean rewriting all five dense/sparse writers around
an iterator API to save one transient array on a path that runs once per commit
with updates, not per document.

The complexity is the same (`O(maxDoc + updates)`, no per-doc rescan, because
the column is filled by one linear read and then indexed directly), and the
allocation is one `Vec` rather than per-doc. **Recorded, not fixed** — revisit
if and when `doc_values.rs`'s write side grows a streaming entry point, which is
also what `writeValuesMultipleBlocks` (b6 #8) will want.

#### F-10 `[INTENTIONAL]` — the `.dvu` encoding stays in the tree, referenced by no index

`write_numeric_updates`/`read_numeric_updates` and their BINARY twins are no
longer written into any index: no `segments_N`, no `.si`, no
`SegmentCommitInfo::files`. They survive because `lucene-search`'s
`soft_deletes::mark_soft_deleted_via_overlay`/`is_soft_deleted_with_overlay`/
`effective_live_docs_with_overlay` are public API built on them, and
`crates/lucene-search` is **c12's** file set — deleting them means a
non-mechanical change in another batch's crate.

The module doc now says so in as many words ("**The delta encoding below is not
an index format** … Do not reach for them when writing a segment: a segment that
references them is a segment real Lucene cannot open"), and c7's semantic tests
for the encoding are unchanged and still green. **Carry-over for c12**: those
three `soft_deletes` functions are the last consumer; with them gone the
encoding, its codec names and its 20-odd tests can go too.

#### F-11 `[INTENTIONAL]` — compound segments and skip-indexed fields are refused by name, not silently mishandled

Two cases `write_field_updates` rejects with a typed error rather than
attempting:

- **A compound-file segment.** Java writes generational files outside the CFS
  but still reads the *base* column and the base `FieldInfos` through the
  compound reader. This port's entire buffered-update path already requires
  loose files (`index_writer::open_segment_for_deletes` opens `.si`/`.tim`/`.doc`
  directly), and no writer path here produces a compound segment, so
  `Error::CompoundSegment` names the situation instead of mis-resolving the base
  column to nothing.
- **A field declaring a doc-values skip index.** Writing a generation means
  running the field back through the doc-values consumer, and this port's
  consumer has no `writeSkipIndex` (b6 #8c). A field whose `.fnm` claims a
  skipper would come back out of a generation without one, and its own
  `parse_meta` would then reject the file it just wrote.
  `Error::SkipIndexUnsupported` refuses up front.

Also here, and deliberately *not* a divergence to fix: Java's
`handleDVUpdates` filters packets by `update.delGen <= maxDelGen` and prunes the
written ones from `pendingDVUpdates`. That bookkeeping stays in
`index_writer.rs`, where this port's packets live (c7's
`apply_packets_to_segment` already resolves and orders them); `field_updates`
receives an already-resolved, already-ordered `(doc, value)` list. Splitting it
that way is what keeps this module free of the reader-pool machinery c7 F-14
recorded as absent.

### Verdict

Swept clean against 10.5.0. F-1 through F-8 and F-12 through F-16 fixed with
tests; F-9 recorded with a cost statement; F-10/F-11 are scope decisions with
reasons. **c7's F-15 is closed.**

---

## `crates/lucene-codecs/src/doc_values_updates.rs`

### Method correspondence (this batch's additions)

| Rust | Java | Verdict |
|---|---|---|
| `merge_numeric_column` | `ReadersAndUpdates.MergedDocValues` over `reader.getNumericDocValues(field)` + `DocValuesFieldUpdates.mergedIterator(subs)` | same result, materialised (F-9) |
| `merge_binary_column` | the BINARY arm of the same | as above |
| `write_numeric_generation` | `Lucene90DocValuesConsumer.addNumericField` via `PerFieldDocValuesFormat` | dense/sparse/empty choice is `writeValues`' |
| `write_binary_generation` | `addBinaryField` | as above |
| `write_numeric_updates` / `read_numeric_updates` / `numeric_value_with_*` (+ BINARY) | — | **not-in-Java**, and no longer an index format (F-10) |

### Findings

Covered by F-1, F-9 and F-10 above; no divergence found in the merge semantics
themselves, which are c7's and unchanged: ascending by doc, later write wins
within a round, `None` is `DocValuesFieldUpdates.reset(doc)` and is distinct
from an empty `BytesRef`.

One property worth naming because the format change makes it *newly*
load-bearing: with delta files, a `reset` and "the base never had a value" were
different bytes; with a full column rewrite they are the same bytes (the doc is
absent from the `IndexedDISI`), which is correct — Java's rewritten column
cannot express the difference either, because there is nothing left underneath
to shadow. The test that used to pin `reset != zero` was rewritten to give the
field a **base column** first, so a reset is now distinguishable from "never had
one" in the only way that still means anything: the base said 7, the generation
says absent, and the neighbouring document's base value survived
(`update_doc_values_with_a_null_value_records_a_removal_not_a_zero`).

### Verdict

Swept. The generational path is new and tested (11 new unit tests); the delta
encoding is unchanged, unreferenced by any index, and documented as such.

---

## `crates/lucene-codecs/src/doc_values.rs`

Swept only for the write-side branch this batch reaches. F-7 above is the one
finding; b6's open items (`writeValuesMultipleBlocks`, the skip-index write,
the LZ4 preset dictionary) are untouched and unchanged.

---

## `crates/lucene-index/src/segment_infos.rs`

### Method correspondence

| Rust | Java | Verdict |
|---|---|---|
| `advance_field_infos_gen` | `SegmentCommitInfo.advanceFieldInfosGen()` | **newly ported** (F-2) |
| `advance_next_write_field_infos_gen` | `advanceNextWriteFieldInfosGen()` | **newly ported** (F-5) |
| `advance_next_write_doc_values_gen` | `advanceNextWriteDocValuesGen()` | **newly ported** (F-5) |
| `set_doc_values_updates_files` | `setDocValuesUpdatesFiles(Map)` | **newly ported**, replace-not-append (F-3) |
| `files` | `SegmentCommitInfo.files()` | unchanged, and now actually exercised: the generational `.fnm` and the three doc-values files are all named there and nowhere else |
| — | `generationAdvanced()`'s `sciId` re-randomisation | still open, and *still* the right shape: c7's A4 recorded it as wanting one pass across every generation-advancing site, and this batch adds two more such sites rather than closing it |

### Verdict

Swept for the generation bookkeeping. c7's A4 (`sci_id` not regenerated when a
generation advances) is unchanged and now has a slightly larger blast radius —
re-recorded in the ledger.

---

## `crates/lucene-index/src/index_writer.rs` — two edits (c10's file)

Kept minimal and mechanical, as the batch brief asked:

1. `write_doc_values_update_generation`'s body is now a five-line delegation to
   `field_updates::write_field_updates`, and the two private helpers it used
   (`doc_values_update_file_name`, `record_dv_update_file`) are deleted.
2. `segment_stats` gained the three-line auto-merge guard (F-8).

Plus test updates, which were not optional: seven of c7's tests asserted on the
`.dvu` files' *contents*. They now assert on the resolved **column** instead,
read back exactly the way `SegmentDocValuesProducer` resolves it — a strictly
stronger assertion (the old ones could not have caught a missing `.fnm`
generation, since they never consulted one). The one test whose *semantics* had
to change is
`successive_doc_values_updates_accumulate_as_separate_generations` →
`successive_doc_values_updates_supersede_each_other_at_a_new_generation`: the
"accumulate" it asserted was an artifact of the delta format. The contract it
existed to protect — the newest write wins — is preserved and now also asserts
that the superseded generation is reclaimed.

`every_generational_file_name_round_trips_through_parse_generation` (c7 F-27's
regression test for the decimal-vs-base-36 bug) is retargeted rather than
dropped: it now walks `gen in 1..=100` over the `.dvm`/`.dvd`/`.dvs` and `.fnm`
generational names and the `.liv` name, so the base-36 property c7 fixed is
still pinned after the format it was fixed on went away.

---

## `crates/lucene-search/src/directory_reader.rs` — one edit (c12's file)

Additive: the generational `.fnm` branch in `SegmentReader::open`, a
`dv_generations: Vec<DocValuesGeneration>` field built there, and the
`doc_values_for_field` accessor. `doc_values_meta`/`doc_values_data` keep
returning the base pair, so every existing caller is unaffected for a field with
no generation, and a caller that has not been taught about generations does not
silently start reading a column that is superseded for one field and not
another. F-6 above is the finding.

---

## Gates

- `cargo fmt --all` — clean.
- `cargo clippy -p lucene-codecs -p lucene-index --all-targets -- -D warnings` — clean.
  (`cargo clippy --workspace` reports two pre-existing failures in other
  batches' live files — `lucene-ffi/src/registry.rs`'s `% 1` and
  `lucene-search/src/highlighter.rs`'s redundant closure — untouched.)
- `cargo test -p lucene-codecs -p lucene-index` — green.
- `cargo test -p lucene-search` — green.
- `scripts/verify-write-path.sh` — **19/19**, up from a confirmed 18/18
  baseline.
- `lucene_index::checksum_verify::verify_directory` and
  `check_index::check_directory` both run clean over an updated index in *both*
  directions (Rust-written and Lucene-written), which is what says the
  generational files are properly referenced from `segments_N` rather than
  merely present.

## New tests

| Where | What |
|---|---|
| `lucene-index/src/field_updates.rs` | 18 unit tests: naming/base-36, whole-column rewrite, second round reading the first as its base, all-reset, no-base-column + attributes, per-field generations, the two-format-instance resolution, failure rollback, and seven error paths |
| `lucene-codecs/src/doc_values_updates.rs` | 11 unit tests for the merge + generation writers, including empty-value-vs-removed and the suffix-must-match-the-header property |
| `lucene-codecs/src/doc_values.rs` | 2 unit tests for Java's `numDocsWithValue == 0` shape (`BinaryReader` is covered by the merge tests above and by both fixtures) |
| `lucene-index/src/index_writer.rs` | 1 new test (auto-merge guard, verified to discriminate) + 7 rewritten onto the real format |
| `lucene-index/tests/doc_values_updates_fixtures.rs` (**new**) | 4 differential tests against Lucene's own three-round-updated index |
| `lucene-search/tests/doc_values_updates_reader_fixtures.rs` (**new**) | 2 reader-level differential tests over the same fixture |
| `fixtures/src/VerifyDocValuesUpdates.java` (**new**) | real Lucene reading a **Rust-written** updated index: every value (including the documents a `reset` round removed), the generation bookkeeping, no leaked generations, `CheckIndex` |

## Tier-2 review (`quality-reviewer`)

Run against the batch's files after the gate was first green, against the
10.5.0 tag. It confirmed the parts most at risk — the naming and suffix
derivation, the base-36 generation, the `meta[-2, 0]` shape, the per-field vs
per-round generation split, the `dvUpdatesFiles` replace semantics and the
`SegmentCommitInfo::files()` coverage — and found **two gating defects and four
advisories worth acting on**. All six are now fixed with tests.

#### F-12 `[CORRECTNESS]` — the per-field codec suffix was hardcoded to the caller's, not read off the field. **Fixed.**

**Java**: `PerFieldDocValuesFormat.getInstance` reads
`PER_FIELD_SUFFIX_KEY` off the `FieldInfo` when `docValuesGen != -1` and reuses
*that* number. The suffix is per format **instance**, not globally `0`.

**We did**: thread `index_writer`'s `per_field_codec_suffix(DOC_VALUES_FORMAT_NAME)`
(`"Lucene90_0"`) through every naming and resolution site, and — worse —
`finish_generation` *overwrote* the field's real suffix attribute with it.

**Consequence**: self-consistent for a port-written segment, and wrong the
moment an update is applied to a segment Lucene wrote whose field sits at a
different suffix — which is the OpenSearch case this port exists for. The base
column would resolve to the wrong file (see F-15) and the new generation would
be named under a component the field's own `.fnm` does not claim.

**Fixed**: `field_per_field_component`/`per_field_component` derive the
component from the field's own `PerFieldDocValuesFormat.format`/`.suffix`
attributes, exactly as the read side (`directory_reader::dv_per_field_suffix`)
and the differential test already did; the caller's value is a fallback used
only for a field that has never had a column, which is the one case nothing on
disk fixes.

**Test**: `a_field_is_resolved_through_its_own_codec_suffix_not_the_first_dvm_in_the_si`
— a segment with **two** doc-values format instances, the field's real column
under `Lucene90_1` and a decoy for another field under `Lucene90_0`; the update
must read the base through `Lucene90_1` and name its generation with it.

#### F-13 `[CORRECTNESS]` — the failure path *rewound* the next-write counters, so a retry could reuse a name the failed attempt wrote. **Fixed.**

**Java**: `advanceNextWriteDocValuesGen()` is applied **on top of** the
already-advanced counter — Java has no snapshot and never rewinds, so
generations are strictly monotonic across attempts.

**We did**: `*sci = snapshot` first, then advance — which for a **multi-field**
round put the counter back at `snapshot + 1` while the failed attempt had
consumed `snapshot..snapshot + k`. The file deletion that follows is
best-effort (`let _ =`), and an I/O failure is exactly the case where it also
fails, so a retry could land on a partial file's name. F-5's own doc comment
claimed the opposite.

**Fixed**: both counters are carried across the restore at
`max(snapshot, current)` before advancing.

**Test**: `a_failure_part_way_through_rolls_the_whole_round_back` now asserts
the retry lands at generation **3** (one field's generation 1 was written before
the failure), not 2 — and the assertion is written as the Java behaviour, not
as a pinned divergence.

#### F-14 `[PERF]` — the column merge called the free `numeric_value`/`binary_value` once per document. **Fixed.**

**Java**: `MergedDocValues` makes one forward pass over the reader's iterator.

**We did**: `(0..max_doc).map(|doc| doc_values::numeric_value(data, entry, doc))`
— and `doc_values.rs`'s own module doc names this as the thing not to do: the
free function re-derives everything per call, so a **sparse** column walks the
`IndexedDISI` block headers from the start of the region on every document.
`O(maxDoc x blocks)` where Java is `O(maxDoc + cardinality)`; at 10M docs that
is ~1.5 billion block-header reads for one update round. This is the same
defect shape b13 fixed in `soft_deletes::effective_live_docs`.

**Fixed**: the numeric side holds a `doc_values::NumericReader` cursor. The
binary side had no cursor type, so this batch adds one —
`doc_values::BinaryReader`, the exact mirror of `NumericReader` (forward-only
`DisiCursor`, rewind on a backwards lookup, nothing allocated), with
`binary_value`'s value-slicing half split out as
`binary_value_at_ordinal` so the two share it rather than diverging.

Recorded rather than fixed: F-9's other half, that the merged column is
materialised into a `Vec` at all. With the cursor in place the cost is one
`Vec` and one forward pass, which is the same complexity as Java — what remains
is the transient allocation, and removing it needs `doc_values.rs`'s five
writers to take iterators instead of slices.

#### F-15 `[CORRECTNESS]` — the base column was located by "first `.dvm` in the `.si`", and a mismatch degraded to an empty column. **Fixed.**

**Java**: a field's column is the one its `PerFieldDocValuesFormat` attributes
name; a segment can legitimately carry several format instances.

**We did**: `si_files.iter().find(|f| f.ends_with(".dvm"))`, and then, if that
meta had no entry for the field, returned `Ok(None)` — "this field has no
column" — from `read_base_numeric`/`read_base_binary`.

**Consequence**: the worst failure mode this format has. The merge would start
from an all-absent column and the new generation would drop **every untouched
document's value**, silently, with a well-formed and checksummed file as the
result. Unreachable through this port's writer today (it writes one instance),
reachable against a Lucene-written index — i.e. reachable in the only
deployment this port has.

**Fixed**: the base `.dvm`/`.dvd` are selected by matching the derived codec
suffix against the field's own component (F-12's fix), a missing match is
`Error::MissingBaseColumn`, and a column that parses but carries no entry for
the field is `Error::MissingBaseEntry` rather than an empty start. A field with
*no* format attributes still legitimately returns `Ok(None)` — that is
`PerFieldDocValuesFormat.FieldsReader`'s "in fieldInfos, but has no docvalues".

**Tests**: `a_field_whose_declared_column_is_missing_from_the_si_is_an_error`,
`a_base_column_that_does_not_describe_the_field_is_an_error_not_an_empty_start`
(a `.dvm` that parses cleanly, describing a *different* known field).

#### F-16 `[MISSING]` — an out-of-range update doc id was silently dropped. **Fixed.**

`merge_*_column` guarded `if doc >= 0 && doc < max_doc { … }`, so a resolver bug
that produced an out-of-range doc became "that update silently didn't happen".
`doc_values::write_single_sparse_numeric_field` already errors on the same
condition; the two now agree. Test:
`an_update_outside_the_segments_doc_range_is_an_error_not_a_silent_drop`.

#### The `reset(doc)` path is now exercised by real Lucene, in both directions

The review's sharpest observation about *testing*: the one new byte shape this
batch added to `doc_values.rs` (F-7's `meta[-2, 0]`, and more generally a
rewritten column that is sparse rather than dense) was covered only by
port-internal round-trips, because neither fixture ever reset a value.

Both now do. `write_doc_values_updates_fixture.rs` adds a fourth round that
removes 40 documents' `val` through `update_doc_values` with a `None` value, and
`VerifyDocValuesUpdates` asserts real Lucene reads those documents back as
having **no value** (a different claim from reading back a wrong one) and that
the column's density is `maxDoc - 40`. `GenDocValuesUpdates.java` does the
mirror with `updateDocValues(term, new NumericDocValuesField("val", null))`, so
the Rust decoder is now checked against a Lucene-written *sparse* update
generation, not only dense ones.

### Advisories recorded, not acted on

| # | What | Why not, and who |
|---|---|---|
| A1 | **`SegmentReader::doc_values_for_field` has no production callers.** Every real doc-values consumer — `lucene-ffi`'s `sort.rs` and `facets.rs`, through `SegmentHandle::dv_meta`/`dv_data` — still reads the *base* pair, so an updated field sorts and facets on the superseded column. The reader-side fix is correct and tested but inert outside tests until those call sites are routed through it. `crates/lucene-ffi` is **c13's** and `lucene-search`'s consumers are **c12's**; the accessor is deliberately additive so neither has to change in this batch. **Carry-over**, with the accessor already in place. |
| A2 | **`check_index` reads the base `.fnm`.** `open_fnm` is unconditional, so this port's `CheckIndex` never sees an updated field's `docValuesGen`, never opens a generation column, and would not notice a `docValuesGen` pointing at a missing file. It is a *coverage* gap, not a false failure: `check_files_exist_and_validate` does verify the generational files' checksums via `commit.files(&si.files)`, and both an updated Rust-written index and an updated Lucene-written one come back clean today. `check_index.rs` is **c9's**. **Carry-over.** |
| A3 | **Mechanical gates the reviewer suggested.** A `disallowed_methods` clippy entry on the free `numeric_value`/`binary_value` (with `NumericReader`/`BinaryReader` as the sanctioned multi-lookup API) would have caught F-14, and this is now its *second* occurrence after b13's `effective_live_docs`. A grep gate on `Lucene90_\d` literals outside `per_field_codec_suffix` would have caught F-12. Both are repo-wide tooling changes, not this batch's files. **Carry-over**, added to the ledger. |

---

## The fixture is itself the specification check

The strongest single piece of evidence in this batch is not a test but the Java
generator's output, which was produced *after* the Rust writer and matches it
name for name:

```text
doc_values_gen=4
field_infos_gen=4
field_infos_files=_0_4.fnm
dv_update_files.2=_0_4_Lucene90_0.dvd,_0_4_Lucene90_0.dvm,_0_4_Lucene90_0.dvs
dv_update_files.3=_0_2_Lucene90_0.dvd,_0_2_Lucene90_0.dvm,_0_2_Lucene90_0.dvs
field_dv_gen.val=4   field_dv_gen.tag=2   field_dv_gen.keep=-1
expected_val=,,,,,, … (30 empty cells) … ,7000,31,7000,33,7000,…
```

Two fields at two different generations (F-4), generations 1 and 3's files
absent because generation 4 superseded them (F-3), a never-updated field still
at `-1` (F-6), and a column whose first 30 documents legitimately have no value
after a `reset` round (F-7) — the four rules this batch had to get right, each
visible directly in what Lucene 10.5.0 writes.

---

## Summary

**18 findings**: 7 CORRECTNESS (all fixed), 7 MISSING (5 fixed, 2 recorded with
named owners), 2 PERF (1 fixed, 1 recorded with a cost statement), 2
INTENTIONAL. Five of the CORRECTNESS findings came from the Tier-2 review, and
three of those five (F-12, F-13, F-15) were defects this batch introduced — the
honest shape of replacing a format wholesale.

| # | Class | What | Status |
|---|---|---|---|
| F-1 | CORRECTNESS | the doc-values-update on-disk format was invented (c7 F-15) | fixed, proven in both directions |
| F-2 | MISSING | no `FieldInfos` generation, so nothing pointed at a generation | fixed |
| F-3 | CORRECTNESS | `dvUpdatesFiles` accumulated where Java replaces | fixed |
| F-4 | CORRECTNESS | one `docValuesGen` per round, where Java takes one per field | fixed |
| F-5 | MISSING | no failure path; a half-written round stranded `SegmentCommitInfo` | fixed |
| F-6 | MISSING | the read path never resolved an update generation at all | fixed |
| F-7 | MISSING | `numDocsWithValue == 0` (`meta[-2, 0]`) was not writable | fixed |
| F-8 | CORRECTNESS | a segment carrying only an update generation was auto-mergeable | fixed |
| F-9 | PERF | the merged column is materialised, not streamed | recorded, cost stated |
| F-10 | INTENTIONAL | the `.dvu` encoding survives as `soft_deletes`' serialization | recorded, c12 |
| F-11 | INTENTIONAL | compound segments and skip-indexed fields refused by name | recorded |
| F-12 | CORRECTNESS | the per-field codec suffix was hardcoded, not read off the field | fixed |
| F-13 | CORRECTNESS | the failure path rewound the next-write counters | fixed |
| F-14 | PERF | the column merge re-derived the DISI cursor per document | fixed (+ new `BinaryReader`) |
| F-15 | CORRECTNESS | the base column was found by "first `.dvm`", and a miss became an empty column | fixed |
| F-16 | MISSING | an out-of-range update doc id was silently dropped | fixed |
| A1 | MISSING | `doc_values_for_field` has no production callers (FFI sort/facets read the base pair) | recorded, c12/c13 |
| A2 | MISSING | `check_index` reads the base `.fnm`, never a generation column | recorded, c9 |

**The contract was checked to *discriminate***, not merely to pass. Two
mechanisms were broken in turn and the evidence watched to fail:

| broken | what failed |
|---|---|
| `merge_numeric_column` starting from an all-absent column instead of the base | `VerifyDocValuesUpdates`: `MISMATCH numeric column density: 130 documents have a value, expected 260` — real Lucene, not this port's reader |
| `segment_stats`' new `doc_values_gen != -1` guard removed | `a_segment_carrying_only_a_doc_values_update_generation_is_never_auto_merged`: "the updated segment must survive every merge round" |

That is the evidence that the two things this batch exists to provide — a
format real Lucene reads, and a segment that survives to be read — are actually
load-bearing rather than passing for some other reason.
