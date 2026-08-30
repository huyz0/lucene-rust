# c3-writer-lifecycle

Follow-up batch opened from two b9 carry-overs: F-11 (no `IndexFileDeleter`, so
nothing ever reclaims an orphan file) and F-10 (no RAM accounting and no flush
trigger, so peak memory is O(everything added since the last commit)).

Files swept / changed:

- `crates/lucene-index/src/index_file_deleter.rs` — **new**
- `crates/lucene-index/src/index_writer.rs`
- `crates/lucene-index/src/indexing_chain.rs`
- `crates/lucene-index/src/lib.rs` (one `mod` line)
- `benchmarks/rust-runner/src/index_bench.rs` (peak-RSS reporting + the A/B knob)

Files changed outside the batch, minimally, because `add_document` became
fallible: `crates/lucene-ffi/src/writer.rs` (one call site, `map_err` +
`?` — b15's file, so the edit is confined to that one expression),
`crates/lucene-search/src/{directory_reader,multi_segment}.rs` and
`crates/lucene-search/tests/index_writer_*_fixtures.rs` (test call sites,
`.unwrap()`), `crates/lucene-index/examples/write_full_segment_fixture.rs`.
`segment_writer.rs` and `segment_infos.rs` needed **no** change: b11's
`SegmentCommitInfo::files()` and b9's `write_pending`/`finish_pending`/
`rollback_pending` were already the right hooks.

Java counterparts compared against (all under
`/home/tuong/work/lucene/lucene/core/src/java/org/apache/lucene/`):
`index/IndexFileDeleter.java`, `util/FileDeleter.java`,
`index/IndexDeletionPolicy.java`, `index/KeepOnlyLastCommitDeletionPolicy.java`,
`index/NoDeletionPolicy.java`, `index/IndexFileNames.java`,
`index/SegmentInfos.java` (`files(boolean)`), `index/IndexWriter.java` (every
`deleter.*` call site), `index/IndexWriterConfig.java`,
`index/LiveIndexWriterConfig.java`, `index/DocumentsWriterFlushControl.java`,
`index/FlushByRamOrCountsPolicy.java`, `index/DocumentsWriterPerThread.java`.

---

## `crates/lucene-index/src/index_file_deleter.rs`

New file. Java: `index/IndexFileDeleter.java` + `util/FileDeleter.java` +
`index/IndexDeletionPolicy.java` and its two stateless implementations.

### Method correspondence

| Rust | Java | Status |
|---|---|---|
| `IndexFileDeleter::open` | `IndexFileDeleter(String[], Directory, Directory, IndexDeletionPolicy, SegmentInfos, InfoStream, IndexWriter, boolean, boolean)` | ported: init-scan, `initRefCount` every index file, load every `segments_N` as a `CommitPoint` and `incRef` it, delete everything left at count 0, `policy.onInit` + `deleteCommits`, then `checkpoint(current, false)` |
| `checkpoint(infos, is_commit)` | `checkpoint(SegmentInfos, boolean)` | identical, both branches |
| `refresh()` | `refresh()` | identical (including reclaiming `pending_segments_N`, which Java's also does and comments on) |
| `delete_new_files(files)` | `deleteNewFiles(Collection<String>)` | identical |
| `set_policy(policy)` | `IndexWriterConfig.setIndexDeletionPolicy` + `revisitPolicy()` | ported as one call (the policies here are stateless, so "set" and "revisit" are the same operation) |
| `inflate_gens(files, infos)` | `inflateGens(SegmentInfos, Collection<String>, InfoStream)` | partial — see F-3 |
| `commit_files(infos, include_segments_file)` | `SegmentInfos.files(boolean)` | ported, plus the `.si` parse Java does not need (see F-9) |
| `si_files_for(sci)` | `SegmentCommitInfo.info.files()` (in-memory in Java) | not-in-Java by necessity; cached per `(segment_name, segment_id)` |
| `inc_ref_all` / `dec_ref_all` | `FileDeleter.incRef` / `decRef(Collection)` | identical, including "decRef every file even on failure, delete the ones that hit zero in one pass" |
| `unrefed_files()` | `FileDeleter.getUnrefedFiles()` | identical |
| `apply_policy()` | `KeepOnlyLastCommitDeletionPolicy.onCommit` + `IndexFileDeleter.deleteCommits()` | identical (the `CommitPoint.delete()` / `commitsToDelete` indirection collapses: with no external policy implementations there is nothing to call back into) |
| `delete_files(names)` | `FileDeleter.delete(Collection<String>)` | identical, including the `segments_N`-first two-pass ordering |
| `ref_count` / `exists` | `FileDeleter.getRefCount` / `exists` | identical |
| `commit_count` / `commit_file_names` | `commits.size()` / `CommitPoint.getSegmentsFileName` | not-in-Java (test/introspection surface) |
| `is_index_file_name` | `IndexFileNames.CODEC_FILE_PATTERN` + the `SEGMENTS`/`PENDING_SEGMENTS` prefix tests + the `write.lock` exclusion | identical semantics, hand-rolled matcher instead of a regex |
| `parse_segment_name` | `IndexFileNames.parseSegmentName` | identical |
| `DeletionPolicy::{KeepOnlyLastCommit, KeepAll}` | `KeepOnlyLastCommitDeletionPolicy` / `NoDeletionPolicy` | ported as an enum — see F-7 |

### Java members with no Rust counterpart

`ensureOpen`/`isClosed`/`close` (no writer open/close lifecycle here — b9's
F-18), `startingCommitDeleted` and the `currentCommitPoint == null` stale-NFS
-listing branch, `deletePendingFiles`/`getPendingDeletions` and the
`Constants.WINDOWS` branch in `FileDeleter.delete` (F-4), `incRef(SegmentInfos,
boolean)`'s separate overload (folded into `checkpoint`), `logInfo`/`InfoStream`,
`CommitPoint`'s `IndexCommit` surface (`getUserData`, `getSegmentCount`,
`getDirectory`, `delete`, `isDeleted`), `forceDelete`,
`deleteFileIfNoRef`, `assertCommitsAreNotDeleted`.

### Verdict

Swept clean for the scope this port has callers for. Open: F-3 (per-segment
half of `inflateGens`), F-7 (`SnapshotDeletionPolicy`), F-9 (the `.si` parse).

---

## `crates/lucene-index/src/index_writer.rs`

Java: `IndexWriter.java`'s `deleter.*` call sites plus
`IndexWriterConfig`/`LiveIndexWriterConfig`/`FlushByRamOrCountsPolicy` for the
flush configuration.

### Method correspondence (delta from b9's table)

| Rust | Java | Status |
|---|---|---|
| `open` | `IndexWriter(Directory, IndexWriterConfig)` | now also builds the deleter (Java line 1167) and applies `inflateGens` |
| `add_document` | `addDocument` | **now `Result<()>`** and auto-flushes (`DocumentsWriterFlushControl` + `FlushByRamOrCountsPolicy.onChange`); still no sequence number (b9 F-7) |
| `add_document_with_custom_freq_terms` | — | same change, not-in-Java |
| `flush` | `DocumentsWriter.doFlush` -> `DocumentsWriterPerThread.flush` -> `IndexWriter.publishFlushedSegment` | **new**: extracted out of `prepare_commit`, plus a non-commit checkpoint and a `refresh()` on the failure path (`DocumentsWriterPerThread.abort()`) |
| `maybe_flush` | `FlushByRamOrCountsPolicy.onChange` | new; document count before RAM, exactly Java's precedence |
| `ram_bytes_used` | `IndexWriter.ramBytesUsed()` | new; different quantity by design (F-8) |
| `set_ram_buffer_size_mb` / `ram_buffer_size_mb` | `LiveIndexWriterConfig.setRAMBufferSizeMB` / `getRAMBufferSizeMB` | new, Java's validation verbatim |
| `set_max_buffered_docs` / `max_buffered_docs` | `LiveIndexWriterConfig.setMaxBufferedDocs` / `getMaxBufferedDocs` | new, Java's validation verbatim |
| `set_deletion_policy` | `IndexWriterConfig.setIndexDeletionPolicy` + `deleter.revisitPolicy()` | new |
| `delete_unused_files` | `IndexWriter.deleteUnusedFiles()` | new |
| `checkpoint_committed` | `IndexWriter.checkpoint()` then `deleter.checkpoint(pendingCommit, true)` | new (F-2); runs *after* the caller installs `self.segment_infos`, so a failed sweep never leaves the in-memory view behind the durable commit |
| `live_infos` | Java's `segmentInfos` *is* the live view | not-in-Java; this port documents `segment_infos` as the last commit, so the uncommitted tail is a separate `flushed_segments` |
| `prepare_commit` | `prepareCommitInternal` | now flushes via `flush()` and folds in `flushed_segments` |
| `finish_commit` | `finishCommit` | now installs the published state, clears `flushed_segments`, and `checkpoint_committed`s |
| `rollback` | `rollbackInternal` | now drops `flushed_segments` and runs `deleter.checkpoint(.., false)` + `deleter.refresh()` |
| `delete_all` | `deleteAll` | now checkpoints, so an uncommitted flush's files are reclaimed immediately (Java line 2859/2868) |
| `update_document` / `delete_documents` / `apply_merge` | `updateDocument` / `deleteDocuments` / `commitMerge` | now `checkpoint_committed` after their commit |
| `new_segment_name` (was `next_segment_name`) | `newSegmentName()` | **now bumps `segmentInfos.counter` as it hands out the name**, which is what Java does (F-1) |
| `build_postings_output` | `FreqProxTermsWriter.flush` | now takes the inverted index **by value** and consumes it (F-6) |

### Verdict

Swept. F-1, F-2 fixed with tests; F-5, F-6, F-8 fixed and measured. Open (all
inherited, none new): b9's F-7 (delete queue), F-12, F-13, F-15, F-16, F-20.

---

## `crates/lucene-index/src/indexing_chain.rs`

Java: `IndexingChain`/`TermsHashPerField`, plus `util/Accountable` for the new
method.

| Rust | Java | Status |
|---|---|---|
| `InMemoryInvertedIndex::ram_bytes_used` | `TermsHash`/`DocumentsWriterPerThread`'s `Counter bytesUsed` (`Accountable.ramBytesUsed`) | **new**; a real byte count over an entirely different in-memory shape (F-8, F-9-of-b9) |

Everything else in the file is unchanged from b9.

### Verdict

Swept. The block-pool redesign stays a carry-over, now with a measured cost
(F-9 below).

---

## Findings

### F-1 `[CORRECTNESS]` — a failed or repeated segment-name handout could reuse a name

**Java**: `newSegmentName()` is
`"_" + Integer.toString(segmentInfos.counter++, Character.MAX_RADIX)` with
`segmentInfos.changed()` in the same synchronized block. The counter advances
*at handout*, unconditionally — a flush that then fails still burns the name.

**We did**: `next_segment_name()` was `&self` and read the counter without
bumping it; the bump happened later, on the `SegmentInfos` each caller was about
to write (b9's F-2 fix, which made the *successful* path correct). Three
consequences remained: (a) `prepare_commit` could only ever produce one segment
per commit, because there was no way to hand out a second name; (b) a
`update_document` that failed after `flush_stored_only_segment` left the counter
untouched, so the retry wrote over the failed attempt's files with
`create_output` (truncating); (c) `execute_merge` had to remember to
`self.segment_infos.counter += 1` by hand after the fact.

**Consequence**: (b) is a real silent-corruption window — small, since it needs
a failure between the flush and the commit, but exactly the window Java's
comment about "we could close, re-open and re-return the same segment name"
describes. (a) blocked automatic flushing outright.

**Resolution — fixed**: `new_segment_name(&mut self)` bumps
`self.segment_infos.counter` as it returns the name, and every caller
(`flush`, `update_document`, `execute_merge`) dropped its own bump. The
after-the-fact bumps in `update_document` and `execute_merge` are gone.

**Test**: `a_failing_update_document_leaves_the_writer_state_unchanged` was
asserting the old behaviour (`segment_infos() == before`, counter included); it
now asserts that everything observable — segments, generation, version, commit
id — is unchanged **and** that `counter` advanced by exactly one, with the
reason in the test body.
`update_document_persists_the_bumped_segment_counter_so_a_reopen_never_reuses_a_name`
(b9's) still passes unchanged.

### F-2 `[CORRECTNESS]` — a commit checkpoint alone never reclaims a merge's sources

**Java**: `commitMerge` calls `IndexWriter.checkpoint()`, which is
`deleter.checkpoint(segmentInfos, false)` — a *non-commit* checkpoint that rolls
`lastFiles` forward to the post-merge segment list. The commit checkpoint
(`deleter.checkpoint(pendingCommit, true)`) happens later, from `finishCommit`.

**We did**: the first version of this batch's `apply_merge`/`delete_documents`/
`update_document`/`finish_commit` called only `deleter.checkpoint(infos, true)`.

**Consequence**: the previous non-commit checkpoint's `lastFiles` still held a
reference on every segment the new commit dropped, so their counts never reached
zero. The merge-lifecycle test caught it immediately: after three commits under a
tight merge policy, `_0`/`_1`/`_2`'s ten files each were still on disk — i.e.
the deleter was installed and the leak was *unchanged* for the single most
important case.

**Resolution — fixed**: `IndexWriter::checkpoint_committed` does Java's two
halves in Java's order (non-commit, then commit), and every commit-writing path
goes through it. The doc comment on it states why the first half is not
redundant.

**Test**: `an_automatic_merge_deletes_its_source_segments_files` asserts on the
directory listing that no `_0.*`/`_1.*`/`_2.*` file survives, that the merged
segment's do, and that all three documents are still readable. This is the test
that found the bug.

### F-3 `[MISSING]` — `inflateGens`' per-segment half

**Java**: after a crash, `inflateGens` pushes `SegmentInfos.generation`,
`SegmentInfos.counter` and each segment's `nextWriteDelGen` /
`nextWriteFieldInfosGen` / `nextWriteDocValuesGen` past the highest value seen in
the directory listing, so a name a crashed session may have written is never
handed out again.

**We do**: `IndexFileDeleter::inflate_gens` ports the `generation` and `counter`
halves (applied by `IndexWriter::open`). The per-segment half is **not** ported:
this port's `SegmentCommitInfo` has no `next_write_*_gen` fields at all —
`crate::term_delete`/`crate::deletes` derive the next `.liv` generation as
`del_gen + 1` directly.

**Consequence**: bounded. The generation a `.liv` would be written at can only
collide with a leftover higher-generation `.liv` from a crashed session, and the
deleter's init sweep deletes exactly those (they are unreferenced). The residual
risk is a session that crashes, is reopened, and whose orphan removal *fails* —
the same residual Java's own comment acknowledges it is guarding against.

**Resolution — recorded, not fixed**: adding three fields to
`SegmentCommitInfo` is an exhaustive-struct-literal change across every
construction site in `lucene-index` and `lucene-search` (the same blast radius as
b9's F-13 `sci_id`), and it would have to land together with that. Recorded in
`docs/sweep/m2/LEDGER.md`, with the note in `inflate_gens`' own doc comment.

### F-4 `[INTENTIONAL]` — no Windows delete-on-close emulation

**Java**: `FileDeleter.delete(String)` swallows `NoSuchFileException` /
`FileNotFoundException` when `Constants.WINDOWS`, because Windows leaves a
deleted-but-still-open file visible in directory listings in a "pending delete"
state; `FSDirectory` keeps a `pendingDeletes` set and
`IndexWriter.deletePendingFiles` drains it with a retry loop, and
`IndexFileDeleter`'s constructor folds `directoryOrig.getPendingDeletions()` into
its "relevant files" set.

**We do**: none of it, and the module doc comment says so in as many words
rather than leaving the omission to be inferred. This port targets Linux, where
`unlink` on an open file succeeds and the name disappears immediately, so there
is no pending-deletion state to model. A `NotFound` from `delete_file` is
therefore a real error here — which is exactly what it is for Java on a
non-Windows platform, so this is Java's behaviour on the target platform, not a
weakening of it.

Called out explicitly because it was requested, and because it is the one place
where "we skipped a whole Java mechanism" is the right answer rather than a gap.

### F-5 `[MISSING]` — `ramBufferSizeMB` / `maxBufferedDocs` / the flush trigger

**Java**: `LiveIndexWriterConfig` holds `ramBufferSizeMB` (default 16.0) and
`maxBufferedDocs` (default `DISABLE_AUTO_FLUSH = -1`), each with two
validations: a non-sentinel `ramBufferSizeMB` must be `> 0.0`, a non-sentinel
`maxBufferedDocs` must be `>= 2`, and disabling *both* is refused.
`FlushByRamOrCountsPolicy.onChange` checks the document count first, then RAM,
and marks a DWPT flush-pending.

**We had**: none of it. `add_document` returned `()` and could not flush.

**Resolution — fixed**: `DISABLE_AUTO_FLUSH`, `DISABLE_AUTO_FLUSH_MB`,
`DEFAULT_RAM_BUFFER_SIZE_MB`, `DEFAULT_MAX_BUFFERED_DOCS` as public constants
with Java's values; `set_ram_buffer_size_mb`/`set_max_buffered_docs` with all
three of Java's validations (`Error::InvalidRamBufferSize`,
`Error::InvalidMaxBufferedDocs`, `Error::BothAutoFlushTriggersDisabled`);
`maybe_flush` with Java's precedence; `IndexWriter::flush` as the public
`DocumentsWriterPerThread.flush` equivalent. `add_document` and
`add_document_with_custom_freq_terms` return `Result<()>` — Java's
`addDocument` throws `IOException` for the same reason.

A flushed-but-uncommitted segment lives in a new `flushed_segments` field
(Java keeps it in `segmentInfos`, which is its *in-memory* view; this facade
documents `segment_infos` as the last commit, so the tail is separate) and is
protected by the deleter's non-commit checkpoint. `prepare_commit` folds them
in; `rollback` discards them and their files.

**Tests**: `max_buffered_docs_flushes_a_segment_without_committing_it` (asserts
on the directory listing that `.fdt` exists and no `segments*` does),
`the_ram_buffer_bounds_the_buffer_no_matter_how_many_documents_arrive` (2000
documents at a 0.01 MB buffer: peak buffered bytes stays near the limit, more
than one segment is produced, and all 2000 documents are readable in order after
the commit), `rollback_discards_auto_flushed_segments_and_deletes_their_files`,
`the_auto_flush_setters_port_javas_validation_exactly`,
`ram_bytes_used_tracks_the_buffered_documents_and_resets_on_flush`.

### F-6 `[PERF]` — three live copies of the segment at flush time (fixed, 6.7x peak RSS)

**Java**: a DWPT holds one bounded working set; `bytesUsed` is what
`ramBufferSizeMB` bounds, and `flush()` streams it out.

**We did**: `prepare_commit` held, simultaneously: every buffered `Document`;
the whole `InMemoryInvertedIndex`; the `Vec<TermPostings>` copy of it that
`build_postings_output` built (which *copied* every position and offset out of
the `Occurrence`s rather than moving them); and every output file as a complete
`Vec<u8>`. With no flush trigger, all of that grew linearly with the number of
documents added since the last commit.

**Resolution — fixed**, three changes:

1. **The trigger** (F-5) bounds the buffer, so peak stops being O(n).
2. **`build_postings_output` consumes the inverted index.** It now takes
   `InMemoryInvertedIndex` by value and iterates `terms` with `into_iter()`, so
   each `(field, term)` key's `Vec<PostingEntry>` — and every `Vec<Occurrence>`
   inside it — is freed as soon as it has been transformed, instead of the whole
   inverted index staying live alongside its `TermPostings` copy. The term bytes
   are moved out of the key `String` (`into_bytes()`) rather than copied.
   Ordering is unchanged: a `BTreeMap`'s `into_iter` still yields each field's
   terms in ascending byte order, which is what `write_fields` requires. The
   per-field dispatch became a `HashMap<&str, usize>` over the config list, one
   pass instead of one `range()` scan per configured field.
3. **`flush` reorders its builders** so norms and term vectors (which only
   *read* the inverted index) run before postings (which consumes it), and the
   inverted index is gone before `flush_stored_only_segment` builds the
   stored-fields copy.

**Measured** (`benchmarks/rust-runner`'s `index-bench`, `--release`, fresh
output directory per run; `LUCENE_RUST_RAM_BUFFER_MB=100000` reproduces the
pre-batch "buffer everything until commit" behaviour on the same binary, so
both arms are the same code except for the trigger). `writer_peak_kb` is
`VmHWM` minus the RSS sampled after the corpus is built and before indexing
starts, so the pre-built document vector is excluded from both arms.

| docs | trigger | writer peak RSS | segments | us/doc |
|---|---|---|---|---|
| 100k | disabled | 442 MB / 432 MB | 1 | 21.7 / 22.2 |
| 200k | disabled | 862 MB / 862 MB | 1 | 30.4 / 29.0 |
| 200k | 16 MB (default) | 131 MB / 128 MB | 8 | 21.0 / 21.2 |

Peak memory with the trigger disabled is exactly linear in document count
(442 MB at 100k, 862 MB at 200k); with it on it is flat. **6.7x lower peak at
200k, and the ratio grows without bound.** Throughput *improved* 29.0 -> 21.0
us/doc (1.4x) in the same window, because the working set now fits in cache.

**The 20k-document number `index-bench` reports by default is unchanged by this
batch**, which is the regression check: at 20k the corpus is ~8 MB, no flush
trips, and both arms run byte-identical code. Measured 19.1 / 20.8 / 20.1
us/doc (trigger disabled) against 19.7 / 21.1 / 20.0 us/doc (default) in one
window, and 18.8-19.6 us/doc across five runs in the quietest window available.
b9 measured 16.6-17.8 us/doc on this shape. The ~10% gap is environmental — c1,
c2 and b15 were compiling and running tests throughout, and the A/B *within* a
single window shows no cost from this batch's changes (the two arms at 20k
exercise the same code and measure the same). The added per-`add_document` work
is one `document_ram_bytes` walk over the document's fields; the added
per-`commit` work is one `.si` parse per segment (cached), a directory listing,
and two checkpoint passes — all of it fixed-per-commit, amortized over 20k
documents.

### F-7 `[INTENTIONAL]` — `IndexDeletionPolicy` is an enum, not a trait

Java models it as an abstract class with `onInit`/`onCommit`, whose
implementations call `IndexCommit.delete()`. The two shipped implementations
that carry no state are `KeepOnlyLastCommitDeletionPolicy` (the default) and
`NoDeletionPolicy`; `SnapshotDeletionPolicy` and
`PersistentSnapshotDeletionPolicy` hand the caller a commit-pinning handle.

`DeletionPolicy` here is an enum over the two stateless ones. A trait with a
single implementation would be a transliteration of an extension point nothing
in this port extends, and the snapshot policies need an `IndexCommit`-shaped
handle type with a lifecycle (`snapshot()`/`release()`) that has no caller here.
The `commitsToDelete` indirection collapses with it: with no external policy to
call back into, "which commits die" is a direct computation.

Recorded rather than hidden: a replication caller that needs snapshots will need
the trait, and the enum is the thing to replace.

### F-8 `[INTENTIONAL]` — `ram_bytes_used` measures the buffered documents, not the inverted form

**Java**: `DocumentsWriterPerThread` inverts each document *as it arrives*, so
its `Counter bytesUsed` measures `ByteBlockPool`/`IntBlockPool` slices,
`BytesRefHash` tables and per-field posting arrays. `ramBufferSizeMB` is a bound
on that, with a constant of 1.0.

**We do**: this port inverts once at flush (b9's F-8 fix, worth 1.74x, and the
reason a single shared pass exists at all), so at `add_document` time the only
structure that exists is the buffered `Document` arena. `ram_bytes_used()`
counts that arena exactly — `Vec<Document>` slots plus every owned
`String`/`Vec<u8>` capacity, accumulated incrementally, never recomputed. It is
a real count, not a sample or an estimate.

**Consequence**: the threshold bounds the buffered-document bytes, and the
transient inverted structure built during the flush is a multiple of it.
Measured on `index-bench`'s corpus (20k docs x 40 tokens from a 20k-word
vocabulary): 8.3 MB of document text becomes 78.5 MB of `InMemoryInvertedIndex`
— **9.4x**. So a 16 MB setting yields a ~130 MB peak on that shape, which is
what the F-6 table shows. Peak memory is bounded by configuration and
independent of document count, which is the property that was missing; the
constant is not Java's.

**Resolution — recorded, with the measurement.**
`InMemoryInvertedIndex::ram_bytes_used()` exists precisely so the constant is
measurable rather than guessed, and both doc comments carry the number. Closing
it needs the per-document incremental invert, which needs a borrowed-token
`Analyzer` API and a byte-pool posting representation — b9's F-9 carry-over,
now quantified.

### F-9 `[PERF]` — where the 9.4x actually goes, and one rejected fix

Breaking down the 78.5 MB: with that much term diversity nearly every
`(doc, term)` pair is unique, so 800k tokens become 800k `PostingEntry`s. Each
costs a `(String, String)` key slot, the `PostingEntry` itself, and a
`Vec<Occurrence>` whose **first `push` reserves capacity 4** — 48 bytes of
allocation for 12 bytes of payload. That surplus alone is ~29 MB of the 78.5.

**Tried and rejected**: `shrink_to_fit()` on the occurrence vector as it moves
from the per-document grouping map into the accumulator (one realloc for the
entries that have surplus, none for the ones that don't). It works on the
structure — 78.5 MB -> 49.8 MB, 9.4x -> 5.98x — but costs **25-60% indexing
throughput** and moves **peak RSS not at all** (862 MB unbounded at 200k, both
with and without it): glibc keeps the freed 48-byte chunks in its arena rather
than returning them, and the reallocs add fragmentation. Reverted, with the
measurement recorded in `ram_bytes_used`'s doc comment so nobody tries it again.

The contained version of this fix is an inline-capacity-1 occurrence
representation, which changes `PostingEntry`'s public shape. Recorded on the
existing block-pool carry-over.

### Also worth recording: the `.si` parse the deleter needs

`SegmentCommitInfo::files()` (b11) needs the segment's `.si`-declared file list,
which Java has in memory on a live `SegmentInfo` and this port must read and
parse (checksum verification included) from disk. The deleter caches it per
`(segment_name, segment_id)` — sound because a segment's `.si` is immutable once
written and a new segment always gets both a new name and a new id — so the cost
is one parse per segment per writer session, not per checkpoint. It does make
`checkpoint` fallible in a way Java's is not: a `SegmentCommitInfo` naming a
segment with no `.si` on disk is now an error rather than being silently
skipped. That is the right direction (such a commit is corrupt), and it did
surface in the existing test suite: four `IndexWriter` tests seeded a writer with
a phantom fixture segment that had no `.si` at all. They now write a real
minimal `.si` and checkpoint it, which is what a real flush does.

---

## Verdicts

- **`index_file_deleter.rs`** — swept clean for the ported scope. Open: F-3
  (per-segment `inflateGens`, blocked on `SegmentCommitInfo` fields), F-7
  (`SnapshotDeletionPolicy`, no caller).
- **`index_writer.rs`** — swept; F-1, F-2, F-5, F-6 fixed with tests, F-8
  recorded with a measurement. No new open item; b9's F-7/F-12/F-13/F-15/F-16/
  F-20 are unchanged.
- **`indexing_chain.rs`** — swept; `ram_bytes_used()` added, the block-pool
  carry-over now carries a number and a rejected fix.

## Gates

- `cargo fmt --all` clean.
- `cargo clippy -p lucene-index --lib --tests --benches -- -D warnings` clean.
- `cargo clippy -p lucene-index --all-targets -- -D warnings` fails on **one
  line in a file this batch does not own**:
  `crates/lucene-index/examples/write_merged_segment_fixture.rs:89`,
  `manual implementation of .is_multiple_of()`. That file is untracked (`??` in
  `git status`) and was created by a concurrently running batch; it does not
  call anything this batch changed except `add_document`, which it already
  handles correctly. Left alone deliberately rather than edited under another
  batch's feet; retried four times over four minutes with no change. Every other
  target in the crate is clean.
- `cargo test -p lucene-index` — all green (425 lib + 16 integration at the
  final run; the lib count also moves with a concurrent batch's additions to
  `merge.rs`). This batch adds **27 tests**: 15 in `index_file_deleter.rs`, 11
  in `index_writer.rs` (10 lifecycle, asserting on the directory listing, plus
  the auto-flush/validation set), and 1 in `indexing_chain.rs`. Four existing
  `index_writer.rs` tests changed: `a_failing_update_document_leaves_the_writer_state_unchanged`
  (F-1) and the three that share `writer_seeded_with_fixture`, which now writes
  a real `.si` for the seeded segment and checkpoints it.
- `cargo test -p lucene-ffi` (439) and `cargo test -p lucene-search` (731 +
  integration) both green, i.e. the `add_document -> Result<()>` signature
  change broke nothing downstream.
- `scripts/verify-write-path.sh` — 14/14 against real Lucene 10.5.0.

## Summary

9 findings:

- **CORRECTNESS 2** — F-1 (a burned segment name could be reused after a
  failure) and F-2 (a commit-only checkpoint never reclaims a merge's sources),
  both **fixed with directory-listing tests**.
- **MISSING 4** — F-11-from-b9 (`IndexFileDeleter`) and F-10-from-b9
  (`ramBufferSizeMB`/`maxBufferedDocs`/flush trigger) **both fixed**, plus
  `IndexWriter.deleteUnusedFiles()` and `setIndexDeletionPolicy` newly ported;
  F-3 (per-segment `inflateGens`) recorded with its blocker.
- **PERF 2** — F-6 (three live copies at flush) **fixed and measured**: writer
  peak RSS 862 MB -> 128 MB at 200k documents, O(n) -> O(1), with throughput
  improving 29.0 -> 21.0 us/doc; F-9 (the 9.4x invert expansion) measured, one
  contained fix tried and rejected on evidence.
- **INTENTIONAL 3** — F-4 (no Windows delete-on-close, stated explicitly), F-7
  (`DeletionPolicy` as an enum), F-8 (`ram_bytes_used` measures a different
  quantity than Java's, with the constant measured).
