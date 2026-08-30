# c36 — merge metadata, the segment-commit change token, zero-doc merges, and the `.si` write

Batch scope: the two remaining wrong-answer findings in `LEDGER.md`'s
"Open work, prioritised" (items **1** and **2**) plus the two adjacent items
living in the same files (**11**, zero-doc merges; **19**, the `.si` rewritten
five times per commit).

Java read from **`/home/tuong/work/lucene-10.5.0`** throughout (the pinned
tag), never from the working tree's `main`.

Files swept:

| Rust file | Java counterpart(s) |
|---|---|
| `crates/lucene-index/src/merge.rs` | `index/SegmentMerger.java`, `index/MergeState.java`, `index/LeafMetaData.java` |
| `crates/lucene-index/src/index_writer.rs` (merge + flush paths) | `index/IndexWriter.java` (`mergeMiddle`, `commitMerge`, `sealFlushedSegment`, `applyAllDeletesAndUpdates`) |
| `crates/lucene-index/src/segment_infos.rs` (`SegmentCommitInfo`) | `index/SegmentCommitInfo.java`, `index/SegmentInfos.java` |
| `crates/lucene-index/src/deletes.rs` | `index/ReadersAndUpdates.java` (`writeLiveDocs`) |
| `crates/lucene-index/src/segment_writer.rs` | `index/IndexWriter.java` (`sealFlushedSegment`), `index/DocumentsWriterPerThread.java` (`flush`) |

New fixture: `fixtures/src/GenMergeMetadata.java` →
`fixtures/data/merge_metadata/`, consumed by
`crates/lucene-index/examples/write_merged_metadata_fixture.rs` and checked by
`fixtures/src/VerifyMergedMetadata.java` (a 23rd `verify-write-path.sh` case).

---

## `crates/lucene-index/src/merge.rs`

Java: `org/apache/lucene/index/SegmentMerger.java`,
`org/apache/lucene/index/LeafMetaData.java`.

### Method correspondence (the part this batch touched)

| Rust | Java | Verdict |
|---|---|---|
| `MergeSource` (struct) | `CodecReader` + `LeafMetaData` as seen by `SegmentMerger` | was **missing** both `LeafMetaData` fields; now carries them |
| `merge_segments` — `SegmentInfo.min_version` | `SegmentMerger`'s constructor, the `minVersion` fold | was **divergent**; fixed |
| `merge_segments` — `SegmentInfo.has_blocks` | `IndexWriter.mergeMiddle`'s `hasBlocks` loop | correct since c22; **moved** onto `MergeSource` |
| `merged_min_version` (new) | `SegmentMerger` ctor lines 74–88 | identical |
| `MergeOptions` | (no Java counterpart — this port's merge parameters) | `has_blocks` removed from it |

### Findings

**1. [CORRECTNESS] The merged `.si` claimed the merging writer's `minVersion`.**

Java (`SegmentMerger`'s constructor):

```java
Version minVersion = Version.LATEST;
for (CodecReader reader : readers) {
  Version leafMinVersion = reader.getMetaData().minVersion();
  if (leafMinVersion == null) { minVersion = null; break; }
  if (minVersion.onOrAfter(leafMinVersion)) { minVersion = leafMinVersion; }
}
segmentInfo.minVersion = minVersion;
```

This port wrote `min_version: Some(lucene_version)` — the version of the
*writer performing the merge* — unconditionally.

*Consequence*: the merged segment claims it was never touched by the older
Lucene whose bytes it is still carrying. Nothing errors: `minVersion` affects
no checksum, no document and no query. It is the single field
`SegmentInfos.readCommit` uses to decide `IndexFormatTooOldException`, and the
one an upgrade tool reads to decide whether a segment must be rewritten. The
finding was latent only because every segment this port had ever merged was
one it had written itself, at its own version — the moment a merge takes in a
segment another Lucene wrote (the entire point of a compatible port) the two
numbers differ.

*Resolution*: **fixed.** `MergeSource` gained
`min_version: Option<LuceneVersion>` (this source's own
`SegmentInfo.minVersion`, i.e. Java's `LeafMetaData.minVersion()`), and
`merge_segments` folds it through the new `merged_min_version`, a line-for-line
port of the loop above including the `null` short-circuit and the
`Version.LATEST` seed. `IndexWriter::execute_merge` fills it from each source's
parsed `.si`, alongside `has_blocks`.

**2. [MISSING → now covered] `has_blocks` was a `MergeOptions` field, not a
per-source one.**

The *behaviour* was already Java's — c22 finding 24 put the
`sources.any(hasBlocks)` disjunction in `IndexWriter::execute_merge` and passed
it down as `MergeOptions::has_blocks`, which is where
`IndexWriter.mergeMiddle` computes it too. So this is **not** a new
wrong-answer; `LEDGER.md` item 1 named it alongside `minVersion` because the
two travel together in Java, and only the `minVersion` half was actually
broken.

It has still been **moved onto `MergeSource`**, because that is the shape Java
gives it: `LeafMetaData(createdVersionMajor, minVersion, sort, hasBlocks)` is
one per-reader record, and every place Java folds `hasBlocks`
(`SlowCompositeCodecReaderWrapper`, `ParallelLeafReader`,
`IndexWriter.addIndexes`) reads it off `reader.getMetaData()`. Keeping the two
halves of one Java record in two different parameters is exactly how the
`minVersion` half went missing for three batches. `MergeOptions::has_blocks`
is gone; `MergeOptions` now holds only the HNSW parameters.

**Blast radius**: 85 exhaustive `MergeSource` struct literals across
`merge.rs` (its test module), `index_writer.rs` (the one production site) and
`benchmarks/rust-runner/src/merge_bench.rs` — c34's estimate of "~85 call
sites" was exact. Seven more use `..MergeSource::stored_only(..)`
functional-update syntax and needed nothing.
`MergeSource::stored_only` defaults both to Java's "the caller told us nothing"
values — `min_version: None` (which makes the merged segment's `null`, by
`SegmentMerger`'s own rule) and `has_blocks: false`.

### Verdict

Swept clean for the two `LeafMetaData` fields. `MergeSource` now carries every
per-reader fact `SegmentMerger` derives the merged `SegmentInfo` from.

---

## `crates/lucene-index/src/segment_infos.rs` + `deletes.rs`

Java: `org/apache/lucene/index/SegmentCommitInfo.java`,
`org/apache/lucene/index/ReadersAndUpdates.java`.

| Rust | Java | Verdict |
|---|---|---|
| `advance_del_gen` | `advanceDelGen()` | was **missing** `generationAdvanced()`; fixed |
| `advance_doc_values_gen` | `advanceDocValuesGen()` | same |
| `advance_field_infos_gen` | `advanceFieldInfosGen()` | same |
| `set_buffered_deletes_gen` | `setBufferedDeletesGen(long)` | same |
| `advance_next_write_*_gen` | `advanceNextWrite*Gen()` | identical (Java deliberately does **not** re-id here) |
| `generation_advanced` (new) | `generationAdvanced()` | ported, minus the `sizeInBytes` cache this port has no analogue for |
| `deletes::apply_deletes` | `ReadersAndUpdates.writeLiveDocs` | was rebuilding the `SegmentCommitInfo` by hand; now mutates it through `advance_del_gen` |

### Findings

**3. [CORRECTNESS] `SegmentCommitInfo.sci_id` was written once and never
changed.**

Java funnels all four generation-advancing mutators through

```java
private void generationAdvanced() {
  sizeInBytes = -1;
  id = StringHelper.randomId();
}
```

and documents the field as "an Id that uniquely identifies this segment commit
… This ID changes each time the segment changes due to a delete, doc-value or
field update". This port stepped the generations and carried the id across
verbatim — `deletes::apply_deletes` even hand-rebuilt the whole
`SegmentCommitInfo` with `sci_id: sci.sci_id`.

*Consequence*: no read breaks — nothing in Lucene validates the bytes
(`SegmentInfos.readCommit` accepts any 16, and a `.si` id is checked against
`SegmentInfo.id`, never against this one). What is lost is the field's only
documented use: as a **change token**. Two commits that differ report the same
id, so `DirectoryReader.openIfChanged`-style per-segment reuse, an NRT or
replication client's "must I re-fetch this segment", and any cache keyed on it
are all told "unchanged" across a delete or a doc-values update.

*Resolution*: **fixed.** `SegmentCommitInfo::generation_advanced` is called
from all four mutators, and `deletes::apply_deletes` now goes through
`advance_del_gen` on a clone of the `SegmentCommitInfo` instead of rebuilding
it (which is also what stopped the id being silently carried across).

The new id is **derived, not random**: a `DefaultHasher` over
`(segment_id, del_gen, field_infos_gen, doc_values_gen, buffered_deletes_gen)`.
This workspace carries no CSPRNG dependency (the same reasoning
`segment_writer::derive_sci_id` and `index_writer::generate_segment_id`
already record), and the only property a consumer reads the id for is
"different whenever the segment-commit is different" — which a hash of the
segment's own id plus all four generations gives: the generations only
increase, so no two states of one segment share an input.

The converse ("same state, same id") is **not** claimed across sessions and the
doc comment says so: `buffered_deletes_gen` is not serialized (neither here nor
in Java), so a segment reaching `delGen = 1` in a fresh writer session gets a
different token than it did in the session that first deleted from it. That is
the safe direction — the consumer is told "changed" and re-fetches — and still
strictly tighter than Java, whose random draw makes every re-derivation look
changed.

### Verdict

Swept clean. Every generation-advancing site is Java's, and the two
`advanceNextWrite*Gen` sites correctly do **not** re-id.

---

## `crates/lucene-index/src/index_writer.rs` — the merge path

Java: `IndexWriter.mergeMiddle` / `IndexWriter.commitMerge` /
`SegmentInfos.applyMergeChanges`.

| Rust | Java | Verdict |
|---|---|---|
| `execute_merge`'s `live_doc_count` guard (new) | `if (merger.shouldMerge()) merger.merge();` + `mergeMiddle`'s `shouldMerge() == false` early return | ported |
| `apply_merge` | `commitMerge(merge, docMaps)` with `dropSegment == false` | now a thin wrapper |
| `drop_merge` (new) | `commitMerge` with `dropSegment == true` | ported |
| `commit_merge` (new, private) | `commitMerge` + `SegmentInfos.applyMergeChanges` | ported |

### Findings

**4. [MISSING] A merge whose result holds no live document was committed.**

Java's `SegmentMerger.shouldMerge()` is `segmentInfo.maxDoc() > 0`, and
`mergeMiddle` guards the whole merge with it:

```java
if (merger.shouldMerge()) { merger.merge(); }
...
if (merger.shouldMerge() == false) {
  // Merge would produce a 0-doc segment, so we do nothing except commit the
  // merge to remove all the 0-doc segments that we "merged":
  success = commitMerge(merge, docMaps);
  return 0;
}
```

`commitMerge` then sets `dropSegment = allDeleted`, and
`SegmentInfos.applyMergeChanges(merge, dropSegment)` removes every merged-away
source from the segment list while inserting nothing in their place; the
merged segment's files are deleted (`deleteNewFiles(merge.info.files())`) —
except that on this path none were ever written.

This port ran the merge and published the empty result.

*Consequence*: a genuine zero-document segment in the commit, whose `.si`,
`.fnm`, `.fdt`/`.fdx`/`.fdm` (and every configured codec file) every later
open, merge and `CheckIndex` pays for, forever.

*Resolution*: **fixed.** `execute_merge` sums the sources' live document counts
right after their stored-fields readers are opened — before any per-format
reader is built, matching where Java's guard sits — and on zero calls the new
`IndexWriter::drop_merge`, which is `commitMerge` with `dropSegment` set:
sources retired, nothing published, no files written. `apply_merge` and
`drop_merge` share one private `commit_merge`, so the public API is unchanged.

**5. [MISSING — recorded, not fixed] 100%-deleted segments are not dropped
when deletes are applied.**

Met while writing finding 4's test, which had to construct fully-deleted
segments by hand precisely because this port never drops them.
`IndexWriter.finishApply` drops them:

```java
if (result.allDeleted() != null) {
  for (SegmentCommitInfo info : result.allDeleted()) { dropDeletedSegment(info); }
  checkpoint();
}
```

with `closeSegmentStates` collecting a segment iff
`segState.rld.isFullyDeleted()` (hard deletes only:
`getDelCount() == info.info.maxDoc()`) **and**
`config.getMergePolicy().keepFullyDeletedSegment(...) == false`.

*Why recorded rather than fixed*, stated plainly because the protocol's default
is to fix a `MISSING` without asking — three things this port does not have:

1. **The policy hook is load-bearing.** `MergePolicyConfig` has no
   `keepFullyDeletedSegment`, and it is not decoration:
   `SoftDeletesRetentionMergePolicy` returns `true` from it. Dropping without
   the hook is correct only for the default policy, and gets *less* correct as
   this port's soft-delete support grows — trading a kept-forever segment for a
   silently-deleted one.
2. **There is no `adjustPendingNumDocs`, and no reader pool** for
   `dropDeletedSegment`'s `mergingSegments` guard to consult.
3. **It has an observable ripple, not a mechanical edit.**
   `a_rollback_after_a_buffered_delete_was_applied_restores_the_committed_segment_list`
   asserts on `writer.segment_infos().segments[0]` *after* deleting that
   segment's only document. With the drop, that list is empty and the test
   indexes out of bounds — so the change needs its own reasoning about what a
   rollback restores, not a fix-up.

Queued in `LEDGER.md` as item **11b** with all three named. It is one contained
change once `MergePolicyConfig` grows the hook.

### Verdict

Zero-doc merges match Java. One adjacent `MISSING` recorded with its blocker
named.

---

## `crates/lucene-index/src/index_writer.rs` + `segment_writer.rs` — the flush path

Java: `IndexWriter.sealFlushedSegment`, `SegmentInfo.files()` accumulating
through `TrackingDirectoryWrapper`.

| Rust | Java | Verdict |
|---|---|---|
| `segment_writer::write_stored_only_segment_files` (new) | `DocumentsWriterPerThread.flush`'s codec writes | ported |
| `segment_writer::seal_flushed_segment` (new) | `IndexWriter.sealFlushedSegment`'s `.si` write | ported |
| `flush_stored_only_segment{,_with_blocks}` | (unchanged public API) | now build + seal |
| `flush_sorted_stored_only_segment` | ditto | second `.si` write removed |
| `IndexWriter::write_{postings,term_vector,doc_values,norms,vector}_files` | each format's `fieldsConsumer`/`Writer` | now return file names |
| `IndexWriter::write_index_sort_to_si` | (no Java counterpart) | **deleted** |

### Findings

**6. [PERF] The `.si` was read, parsed, extended, rewritten and fsynced once
per file group.**

Java accumulates `SegmentInfo.files` in memory as each format writes, and
`sealFlushedSegment` writes the `.si` **once**, at the end. Here, a fully
configured commit wrote it **seven** times: once in
`flush_stored_only_segment_with_blocks`, then once each for postings, term
vectors, doc values, norms and vectors, then once more for the index-sort
descriptor — each of the last six preceded by an `open` + `segment_info::parse`
of the file it was about to overwrite, and each followed by its own
`dir.sync`. Only the last write survived; the six before it were pure I/O.
`flush_sorted_stored_only_segment` did the same twice.

*Resolution*: **fixed.** `segment_writer` gained `FlushedSegment` (the
in-memory `SegmentInfo` + the `SegmentCommitInfo` + the names still to fsync),
`write_stored_only_segment_files` (everything up to but not including the
`.si`) and `seal_flushed_segment` (the single `.si` write plus one fsync of
the whole set). The five `write_*_files` helpers now just write their files and
return their names; `IndexWriter::flush` pushes those into
`flushed.info.files`, sets `index_sort` on the same in-memory struct, and
seals once. `write_index_sort_to_si` is gone.

**Measured.** The wall-time saving is **below the noise floor on this host**,
and that is the honest result rather than a disappointing one. The `.si` for
these segments is a few hundred bytes and the container's filesystem does not
make `fsync` expensive: with a temporary hook adding *n* extra
read-modify-write-then-resync cycles per flush, 50 000 documents at
`RAM_BUFFER_MB=0.5` (57 flushed segments) measured

| extra cycles per flush | µs/doc (3 runs) |
|---|---|
| 0 | 15.18 / 15.75 / 15.20 |
| 2 | 15.28 / 15.67 / 14.90 |
| 6 | 15.57 / 14.96 / 15.62 |

and pushing it to 20 000 extra cycles on a single flush (20 000 documents,
written to the bind-mounted host filesystem rather than the container tmpfs)
still produced no separation: 24.34 / 22.95 µs/doc at *n*=0 against
21.77 / 21.41 at *n*=20 000. One cycle therefore costs well under a
microsecond here — `File::sync_all` on this WSL2 mount is effectively free —
so the change does not move the ~19.9 µs/doc indexing baseline (measured after
the change: 18.70–20.62 µs/doc over five runs, 2 flushed segments; the run-to-run
spread is larger than anything this could contribute).

What the change *does* remove is exact and countable, and it is what matters on
a filesystem that honours `fsync` — the durability contract this code is
written against: per flushed segment, **4 file writes, 4 fsyncs and 4
`open`+`parse` round trips** in the configuration the new test exercises (six of
each in the maximum case, with vectors and an index sort), and **1 write + 1
fsync** per sorted flush in `segment_writer`. That is asserted directly rather
than timed, by `one_commit_writes_the_segments_si_exactly_once`, which drives
the writer through a counting `Directory` and requires exactly one
`create_output("_0.si")` and one `sync`.

**7. [PERF — recorded] `IndexFileDeleter`'s checkpoint re-reads the finished
`.si`.**

The one remaining `.si` read per flush. Java's `IndexFileDeleter.checkpoint`
reference-counts from the in-memory `SegmentCommitInfo.files()`, which already
holds the `SegmentInfo`; this port re-opens and re-parses the file. One read
per segment per checkpoint, against Java's zero. Not fixed here: it needs the
deleter to be handed the live `SegmentInfo`s rather than segment *names*, which
is a signature change across `index_file_deleter.rs` and every checkpoint call
site — a batch of its own, and now unblocked, since `FlushedSegment` is the
in-memory `SegmentInfo` that call would need. Recorded in `LEDGER.md`.

### Verdict

`.si` written once per flushed segment, as in Java. One adjacent `PERF` item
recorded and unblocked.

---

## Hygiene, met not sought

### `scripts/gen-fixtures.sh --check` was failing

`fixtures/data/break_iterator/manifest.properties` recorded
`java_version=25`, and the container pins **JDK 21** — the toolchain the gate
and every other fixture are defined against. So one committed fixture had been
generated on the host, outside `scripts/docker-test.sh`, which is exactly what
that container exists to prevent. Only the `java_version` line differed (the
eleven break-iterator boundary sets are identical on 21 and 25), so nothing
downstream was wrong — but `--check` was red, and a red check is one nobody
runs.

**Fixed** by `scripts/gen-fixtures.sh --only GenBreakIterator` inside the
container. `--check` is now green: 48 deterministic files byte-identical, 0
mismatches, 0 missing, 0 unexplained extras, 0 manifests with a wrong key set,
0 segment-id baseline disagreements — including this batch's new
`merge_metadata/`.

### `benchmarks/rust-runner` did not compile

Met, not sought: `cargo build --release` in `benchmarks/rust-runner` failed on
two half-finished edits left by an earlier batch's API reshapes —
`merge_bench.rs` used `MergeSortKeySpec { field, reverse, missing }` fields
that `IndexSortField` replaced (and did not import `IndexSortField`), and
`main.rs` used `DocValueSegment::doc_values_data` where the struct now has
`range_data`/`sort_data`. The crate is outside the workspace, so neither
`cargo clippy --workspace` nor the gate sees it, and the stale binary left in
`target-docker/release/` kept producing plausible benchmark numbers from
pre-change code — the first `index-bench` numbers taken for finding 6 came from
that stale binary and had to be thrown away. **Fixed** (three small edits); the
crate builds again, which is what made finding 6's measurement possible at all.

Both are the same shape: a check that exists but is not in the gate, so it goes
quietly red and stays that way. Worth a `LEDGER.md` tooling entry on its own
(it is adjacent to item 26's "a rustdoc pass belongs in the gate").

---

## Differential evidence

`scripts/verify-write-path.sh` case 23, `VerifyMergedMetadata <-
write_merged_metadata_fixture`:

- `fixtures/src/GenMergeMetadata.java` writes three segments through a real
  `IndexWriter` (`NoMergePolicy`, no compound files) — `_1` built with
  `addDocuments`, so **Lucene itself** sets its `hasBlocks` — then rewrites
  each segment's `.si` through `codec.segmentInfoFormat().write` with a chosen
  `minVersion` (10.2.0, **10.0.0**, 10.1.0; oldest in the middle, so a fold
  that takes the first or the last source is caught), carrying name, id,
  files, codec, diagnostics, `maxDoc` and `hasBlocks` across unchanged. It
  re-reads the commit afterwards to prove the rewritten `.si` files still form
  an index Lucene opens.
- `write_merged_metadata_fixture.rs` copies that index, opens it with this
  port's `IndexWriter` at version **10.5.0** (newer than every source, which
  is what makes "the writer's version" and "the minimum over the sources"
  distinguishable at all), and merges all three.
- `VerifyMergedMetadata.java` reads `LeafMetaData.minVersion()` and
  `hasBlocks()` off the merged segment, plus every document's stored `id` in
  merged order, plus `CheckIndex` at `MIN_LEVEL_FOR_SLOW_CHECKS`.

**Verified to fail against the unfixed code**, both fields at once:

```
MISMATCH minVersion: merged segment reports 10.5.0 expected 10.0.0
MISMATCH hasBlocks:  merged segment reports false  expected true
```

This is the shape the ledger asked for: it can only pass if the fold is real,
because a port that copies the writer's own version produces an index that
opens cleanly, reads every document correctly and passes `CheckIndex`.

Fixture generated with `scripts/gen-fixtures.sh --only GenMergeMetadata`;
`fixtures/segment-ids.txt` gained exactly four lines and lost none.

## Tier-2 review

Run before declaring the batch done, per AGENTS.md. **Two gating findings, both
fixed, and one design hazard this batch had introduced.**

1. **The `merged_min_version` unit test tested a copy of the function, not the
   function.** It declared a local `fn fold(..)` that restated the loop and
   asserted against that; only the empty-sources case reached the shipped code.
   Fixed by extracting the loop as `fold_min_version(latest, mins)`, which
   `merged_min_version` now delegates to and the test drives directly.
   **Mutation-checked afterwards**: inverting the comparison fails it, dropping
   the `None` short-circuit fails it, ignoring the sources entirely fails it.
   (`>=` → `>` does *not* fail it, and correctly so — the assignment is
   idempotent on equality, so that mutation is behaviourally equivalent.)

2. **A doc comment was attached to the wrong item.** The
   `IndexWriter.mergeMiddle` rationale written for the zero-doc merge test had
   been glued onto `struct CountingDirectory`, leaving the test helper
   documented as being about zero-doc merges and the zero-doc merge test with
   no rationale at all. Moved.

3. **`MergeSource::stored_only` defaulted `min_version` to `None`, which is an
   unopenable index by omission.** This was a real regression introduced by
   finding 1: before it, a merged `.si` always carried *some* version (the
   wrong one); after it, any caller of the public
   `merge_stored_only_segments`/`merge_segments` who did not set the field
   produced a merged segment with **no** `minVersion` — which real Lucene
   refuses for an index created at major 7 or later, in three places
   (`SegmentInfos.readCommit` throws `CorruptIndexException`; `SegmentInfos.write`
   and `applyMergeChanges` throw `IllegalStateException`). Worse, the batch's own
   test had enshrined it as a "control" (`assert_eq!(si.min_version, None)`).
   Note the asymmetry with Java: there, a `null` `LeafMetaData.minVersion()` is
   unreachable in a ≥7 index; here the *default* made it the common case.
   **Fixed** by making `min_version` a required fourth parameter of
   `stored_only` (66 call sites), with the constructor's doc comment stating
   exactly why that one field is not defaulted when every other is. The
   "control" now asserts the merged `.si` carries the writer's version, and a
   separate case covers Java's genuine `null`: one source recording no
   `minVersion` makes the merged segment's `null` too.

Four further advisories, all taken:

- The zero-doc guard's comment claimed it sat "before any format is opened",
  which was false — the `opened` loop has already read every source's files by
  then. Rather than soften the comment, the placement is now *argued*: hoisting
  the guard above that loop would skip the `.fdm`-against-`.si` `maxDoc` check,
  the `.liv` parse, `validate_index_sort` and `check_format_coverage`, so a
  corrupt or unmergeable source would be **silently dropped from the commit**
  instead of reported. Java's guard is after reader construction for the same
  reason.
- `commit_merge` reported `PreparedCommitPending("apply_merge")` for both
  callers; it now takes the caller's name, so `drop_merge` names itself.
- `VerifyMergedMetadata` hardcoded the three expected answers as constants with
  "must match `GenMergeMetadata`" comments, while the generator was already
  writing them into `manifest.properties` (computed from the versions it
  actually stamped, using Java's own `Version.onOrAfter`). Editing the
  fixture's version list would have left the verifier checking stale constants.
  It now reads the manifest, which is how ten of this suite's verifiers already
  work; `verify-write-path.sh`'s existing `extra`-argv field passes the fixture
  directory.
- `benchmarks/rust-runner` being silently broken is a check, not a judgment:
  `cargo check --manifest-path benchmarks/rust-runner/Cargo.toml --all-targets`
  is now a **gate step** (`scripts/gate.sh`, and AGENTS.md's table), so the next
  API reshape fails the gate instead of leaving a stale binary producing
  plausible numbers. Recorded in `LEDGER.md` as item 26b, half-closed.

Also noted and answered in the doc comments: `set_buffered_deletes_gen` runs on
every flush's publish step, so `segment_writer::derive_sci_id`'s value never
survives an `IndexWriter` flush — Java has the same two-step
(`StringHelper.randomId()` in the constructor, re-drawn by
`setBufferedDeletesGen`), and both derivations now say so; and the distinctness
argument in `generation_advanced` establishes distinct *inputs*, with the
distinct-*outputs* bound being the 128-bit digest width, which the comment now
states along with `DefaultHasher`'s cross-release instability being irrelevant
(ids are only ever compared against ids read back out of `segments_N`).

## Tests added, each verified to fail against the unfixed code

| Test | Fails without |
|---|---|
| `segment_infos::every_generation_advance_gives_the_segment_commit_a_new_id` | finding 3 |
| `segment_infos::successive_generations_of_one_segment_get_distinct_ids` | finding 3 |
| `segment_infos::stepping_only_the_next_write_counter_keeps_the_id` | (control: Java's `advanceNextWrite*Gen` must **not** re-id) |
| `deletes::each_round_of_deletes_gives_the_segment_commit_a_new_id` | finding 3 |
| `index_writer::a_delete_changes_the_segment_commit_id_the_next_commit_records` | finding 3, end to end through `segments_N` |
| `merge::merged_min_version_is_the_smallest_source_version_or_none` | finding 1 |
| `merge::a_merged_si_takes_the_oldest_source_min_version_and_any_sources_blocks` | findings 1 and 2 |
| `index_writer::a_merge_whose_sources_are_all_deleted_is_dropped_not_committed` | finding 4 |
| `index_writer::one_commit_writes_the_segments_si_exactly_once` | finding 6 |
| `VerifyMergedMetadata <- write_merged_metadata_fixture` | findings 1 and 2, against real Lucene |
