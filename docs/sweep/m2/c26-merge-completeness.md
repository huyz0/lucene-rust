# c26 — merge completeness: the gate c22's review asked for, and the two carry-overs behind it

Follow-up batch. c22 fixed eleven correctness defects in the merge path; its
Tier-2 review then traced **four** of them to one structural gap, in c22's own
words:

> nothing mechanically checks that `execute_merge` opens every format the flush
> can write

The four were c22 findings **14** (norms — every merged BM25 score wrong, a
carry-over standing since c4), **22** (every doc-values type except NUMERIC),
**23** (positional postings) and **24** (`has_blocks`). None is a crash. Each
produced a merged segment that is well-formed, checksummed and
`CheckIndex`-clean, whose data is simply gone or wrong, and each was found by
reading rather than by a check. This batch builds the check, runs it, and
closes the two merge carry-overs that were blocked behind the same reasoning.

Java read from **`/home/tuong/work/lucene-10.5.0`**, the pinned tag.

| Rust file | Java counterpart (10.5.0) |
|---|---|
| `crates/lucene-index/src/merge.rs` | `index/SegmentMerger.java`, `codecs/DocValuesConsumer.java`, `codecs/NormsConsumer.java` |
| `crates/lucene-index/src/index_writer.rs` (`execute_merge`, `segment_stats`, `set_norms_field`, `build_norms_output`) | `index/IndexWriter.java` (`mergeMiddle`, `updatePendingMerges`), `index/MergeState.java`'s reader assembly, `index/IndexingChain.java`'s norms consumer |
| `crates/lucene-index/src/merge_policy.rs` | `index/TieredMergePolicy.java` |
| `crates/lucene-index/src/field_updates.rs` (two `fn`s made `pub(crate)`) | `index/IndexWriter.readFieldInfos(SegmentCommitInfo)`, `index/SegmentDocValuesProducer` |

Findings: **3 CORRECTNESS** (all fixed, all with negative controls), **4
MISSING** (3 fixed, 1 recorded with a named blocker), **1 PERF** (measured),
**3 INTENTIONAL**. The gate found **one** defect nobody knew about (finding 2)
and confirmed the absence of others across the whole existing merge corpus,
which is reported as the real result it is.

`scripts/verify-write-path.sh`: **22/22**, unchanged — this batch adds no
fixture. Run, not assumed.

---

## The gate

### What it is

`merge::SegmentFormat` — one enum variant per per-segment Lucene format,
carrying the two facts a merge needs about it — plus
`merge::check_format_coverage`, called from `IndexWriter::execute_merge` on the
real path before a byte is merged:

```rust
pub enum SegmentFormat {
    StoredFields, TermVectors, Postings, DocValues, Norms, Points, KnnVectors,
}
impl SegmentFormat {
    pub fn extensions(self) -> &'static [&'static str];   // exhaustive match
    pub fn is_opened(self, source: &MergeSource<'_>) -> bool; // exhaustive match
}
```

The rule it enforces, per source: **every format that source's own
`SegmentCommitInfo::files` lists files for must be a format the caller opened
onto that source's `MergeSource`.** Anything else is
`Error::MergeFormatNotOpened { segment, format, files }`, and the merge is
refused rather than performed — because the loss is otherwise unobservable, and
a merge that runs is a merge whose sources get deleted.

### Why this mechanism, and not the three obvious alternatives

The brief asked for something that cannot rot. Four candidates:

| candidate | why not |
|---|---|
| a hand-maintained list of formats in a `scripts/` checker | precisely the thing that rots; a new format is added by someone who does not know the list exists |
| a static scan of `index_writer.rs` for file-extension literals | a purely lexical guess at which extension belongs to which write path, with no way to tell "opened" from "mentioned" — and the flush names its files through `per_field_segment`, not as literals |
| a check in `check_index` comparing source and merged `.fnm`s | c22 suggested this. It cannot run: `check_index` sees one directory at one moment, and after a merge commits, the sources are **deleted** — the comparison has no left-hand side |
| **a runtime check driven off the source segments' own file lists** | chosen |

Two properties make the chosen one hard to rot, and neither is a list of "the
formats we have today":

1. **An unclaimed extension is an error.** Any extension in a source's file
   list that no `SegmentFormat` claims, and that is not one of the three named
   non-format extensions (`si`, `fnm`, `liv` — each with its reason in the
   table), fails with `Error::UnknownSegmentFormat`. So a flush path that
   learns to write a **new** format is caught by this gate on the very first
   merge of a segment it wrote, *before* anything can be dropped. The only way
   to satisfy that error is to add a `SegmentFormat` variant — which forces
   `extensions()` and `is_opened()` (two exhaustive `match`es) and `ALL` (a
   fixed-length array), and `is_opened` cannot be written without a
   `MergeSource` field to read, which cannot be populated without the caller
   opening the format. That is the closed loop.
2. **It runs on every merge, not in one test.** Being on the real
   `execute_merge` path means every existing and future merge test exercises it
   over whatever formats that test's segments happen to have — the whole
   `lucene-index` suite, not one test.

The source of truth is `SegmentCommitInfo::files(si_files)`, not
`SegmentInfo::files`: a generational `.dvm`/`.dvd` is never listed in the `.si`
(it did not exist when the `.si` was written), and that is exactly the door
c22's finding 18 had to bar with a `segment_stats` exclusion. Reading the
wider list is what let finding 5 remove that exclusion instead.

### What it would miss

Stated plainly, because a gate whose blind spots are unstated is worse than no
gate:

- **That an opened format is merged *correctly*.** A format whose doc map is
  wrong produces exactly the same well-formed segment. That is what c22's
  `assert_every_format_agrees` and c17's `VerifySortedSegment` are for. This
  gate answers "opened at all", which is the question c22 got wrong four times.
- **A format the flush writes without listing it in the `.si`.** Such a file is
  already invisible to `IndexFileDeleter` and to `CheckIndex`, so it is a
  different and louder defect — but this gate would not see it.
- **A format opened per source but dropped per *field*.** c22 finding 22 was a
  per-type drop *inside* an opened `.dvm`. The `debug_assert!` in
  `execute_merge` covers the doc-values case (and earned its keep — it is what
  fired first in finding 5's negative control); the coarser "not opened at all"
  is this gate's job.
- **A format only reachable under a configuration no test uses.** The gate only
  asks about formats a source segment actually has. The companion test
  `every_format_the_flush_writes_reappears_in_the_merged_si` closes this by
  asserting that the sources it builds carry *every* `SegmentFormat` except
  `Points`, so a format silently stopping being exercised fails that assertion
  rather than quietly reducing the gate's reach.
- **Points.** No flush path can write them (`docs/parity.md`), so no segment
  can carry them. The gate covers them anyway and would fire the day one can —
  which is the point.

### Running it

**It found one defect nobody knew about, and confirmed the absence of others.**

Running the gate over the entire existing `lucene-index` test suite (every
merge test c4/c8/c17/c22/c23 wrote) produced **no**
`MergeFormatNotOpened` and **no** `UnknownSegmentFormat`. That is a real and
reportable result: c22's four instances were the four instances, and every
format the flush can write today is opened by `execute_merge` today. It is
worth saying because the alternative — inventing a fifth — would be worse than
finding nothing.

What the gate *did* find is one level down, and it found it by making the two
carry-overs closable: with the `.si`-only blindness removed, `execute_merge`
was reading the **base** doc-values column for a field whose current column is
a generation (finding 4), and the merge then refused a source with no column at
all where Java writes it as missing (finding 2).

---

## crates/lucene-index/src/merge.rs

Java counterparts: `index/SegmentMerger.java`, `codecs/DocValuesConsumer.java`
(`mergeNumericField`, `getMergedNumericDocValues`), `codecs/NormsConsumer.java`
(`mergeNormsField`).

### Method correspondence

| Rust | Java | Verdict |
|---|---|---|
| `SegmentFormat` (new) | — | not-in-Java; Java has no equivalent because `SegmentMerger.merge()` calls a fixed sequence of consumers over a `MergeState` built from `SegmentReader`s, so "the caller forgot to open a format" is not expressible there. This port's `MergeSource` is caller-supplied by design (b10), which is what makes the failure possible and therefore the check necessary. |
| `check_format_coverage` (new) | — | not-in-Java, same reason |
| `merge_norms` | `NormsConsumer.mergeNormsField` | **widened to every field (finding 3)**; was one field per merge |
| `merge_numeric_doc_values` | `DocValuesConsumer.getMergedNumericDocValues` | **divergent (finding 2)**, now matches: a source with no column contributes no values instead of failing the merge |
| `describe_written_files` | `IndexWriter::fields_with_per_field_attributes`' merge-time twin | `norms_field_number: Option<i32>` -> `norms_field_numbers: &[i32]` |
| `merge_{binary,sorted,sorted_numeric,sorted_set}_doc_values` | the same `DocValuesConsumer.getMerged*` family | **still divergent (finding 7, recorded)** — the writer has no sparse shape for these four |

### Findings

1. **[MISSING -> fixed]** *Nothing mechanically checked that `execute_merge`
   opens every format the flush can write.* c22's carry-over, verbatim, and the
   root of its findings 14, 22, 23 and 24. `merge::SegmentFormat` +
   `merge::check_format_coverage`, called from `execute_merge`. See **The
   gate** above for the mechanism, why it was chosen over three alternatives,
   and its five stated blind spots.
   *Tests*: `no_two_segment_formats_claim_the_same_extension` (the extension
   table must partition — two formats sharing one extension would attribute a
   file to the wrong `is_opened` question, and a format claiming `si`/`fnm`/
   `liv` would demand readers for the segment's own bookkeeping);
   `segment_format_all_lists_each_variant_once` (the fixed-length array makes a
   *short* `ALL` a compile error but not a duplicated one, and every extension
   round-trips through `for_extension`);
   `a_format_the_si_lists_and_the_merge_never_opened_is_refused` (c22 finding
   14's exact shape, against real flushed files);
   `every_optional_format_is_refused_when_the_si_has_it_and_the_source_does_not`
   (one case per format, so no `is_opened` arm can be wrong in the permissive
   direction without a test noticing);
   `an_extension_no_format_claims_is_refused_rather_than_ignored` (the anti-rot
   arm, including `.cfs` and a file with no extension at all);
   `the_gate_asks_the_question_per_source` (the "opened it for source 0 and
   forgot the loop" bug shape).

2. **[CORRECTNESS -> fixed]** *A merge failed where Java writes a missing
   value.* `merge_numeric_doc_values` raised
   `Error::DocValuesFieldMissingInSource` when a source contributing live
   documents had no NUMERIC column for a merged field. Java does not:
   `DocValuesConsumer.getMergedNumericDocValues` builds its `subs` list as
   ```java
   FieldInfo readerFieldInfo = mergeState.fieldInfos[i].fieldInfo(mergeFieldInfo.name);
   if (readerFieldInfo != null && readerFieldInfo.getDocValuesType() == DocValuesType.NUMERIC) {
     values = docValuesProducer.getNumeric(readerFieldInfo);
   }
   if (values != null) { subs.add(new NumericDocValuesSub(mergeState.docMaps[i], values)); }
   ```
   — a reader with no column is simply **not a sub**, so every one of its
   documents comes out of the merge with no value. That is the sparse column
   `SortField.setMissingValue` already models and `write_dense_fields` already
   writes (c22 finding 6).
   The consequence is not a wrong segment but two ordinary cases that could not
   be merged at all, each propagating out of `commit`: a field added to the
   schema after some segments were flushed, and a doc-values **update** against
   a field whose base flush wrote no column — the second is finding 5's
   blocker, and is how this surfaced.
   What makes it safe to stop treating this as a caller error is finding 1:
   `check_format_coverage` refuses a merge whose caller never opened a format a
   source's `.si` lists, and the `debug_assert!` in `execute_merge` pins that
   every entry an opened `.dvm` declares reaches the merge. A silently-dropped
   column can no longer hide behind this branch, so the branch can have Java's
   meaning. The other four types keep the error (finding 7).
   *Test*: `numeric_doc_values_absent_from_one_source_come_back_missing_not_as_an_error`
   — the rewritten rejection test, now reading both documents' values back
   through the unmodified reader stack (`Some(10)`, `None`) rather than
   asserting an error.

3. **[MISSING -> fixed]** *Norms took one field per merge* — c22's finding 13
   and open carry-over. `Error::TooManyNormsFields` is gone; every merged norms
   field now goes into one `.nvm`/`.nvd` pair through
   `lucene_codecs::norms::write_fields`, in ascending merged-field-number order
   so the meta entry order is a function of the schema rather than of which
   source was first. That is what a real `Lucene90NormsFormat` segment is —
   `Lucene90NormsConsumer` gets one `addNormsField` call per field into one
   pair — and it is the same widening c22's finding 5 made for doc values.
   **c22's stated blocker was stale.** Its carry-over reads "Needs a
   `norms::write_dense_fields` the way doc values got one... Closing it is a
   `lucene-codecs` change". `norms::write_fields` already existed, from
   `b6-docvalues`, with `write_single_dense_field` a one-element wrapper over
   it (`docs/parity.md`'s norms write-side row records it). So there was no
   `lucene-codecs` handoff to make and the whole carry-over was closable in
   `lucene-index`; the brief's instruction to record a handoff rather than edit
   `crates/lucene-codecs` did not bind, because **no `lucene-codecs` file was
   touched by this batch**.
   *Tests*: finding 4's end-to-end test, plus every pre-existing single-field
   norms test (the one-field path is unchanged, and
   `write_single_dense_field`'s callers were not touched).

7. **[MISSING, recorded]** *BINARY / SORTED / SORTED_NUMERIC / SORTED_SET still
   fail where Java writes missing.* Finding 2 fixed NUMERIC only.
   `DocValuesConsumer` has the identical `values == null` skip in
   `getMergedBinaryDocValues`, `getMergedSortedDocValues`,
   `getMergedSortedNumericDocValues` and `getMergedSortedSetDocValues`, so all
   four diverge the same way. **The blocker is named and is in
   `crates/lucene-codecs`, which `c24-arith-codecs` owns**:
   `doc_values::DenseField` has exactly one sparse variant, `SparseNumeric`,
   added by c22 for the multi-tier index sort. Expressing an all-missing
   BINARY/SORTED/SORTED_NUMERIC/SORTED_SET column needs a sparse variant apiece
   — the `IndexedDISI` + values-body shape `write_single_sparse_numeric_field`
   already has, generalised. See **Handoffs** for the precise request.
   Reachability: a *sorted*-doc-values field that some segments lack is the
   normal case for a schema that grew, so this is not exotic; it is simply not
   what this batch's carry-overs were.

### Verdict

Swept clean for the merge-completeness question. Open: finding 7.

---

## crates/lucene-index/src/index_writer.rs

Java counterparts: `IndexWriter.mergeMiddle` + `MergeState`'s reader assembly
(`execute_merge`), `IndexWriter.updatePendingMerges` (`segment_stats`),
`IndexWriter.readFieldInfos(SegmentCommitInfo)` (the generational read),
`IndexingChain`'s per-field norms consumer (`add_norms_field`).

### Method correspondence

| Rust | Java | Verdict |
|---|---|---|
| `execute_merge` | `IndexWriter.mergeMiddle` + `MergeState`'s constructor | **gated (finding 1)**; doc-values columns now resolved per field to the newest generation (finding 4) |
| `read_sort_keys` | `IndexSorter.LongSorter.getComparableValues` | takes the resolved columns instead of a raw `.dvm`/`.dvd` pair |
| `segment_stats` | `updatePendingMerges` -> `MergePolicy.findMerges` | **the last exclusion dropped (finding 5)**; nothing is withheld |
| `set_norms_field` | `FieldType.setOmitNorms(false)` on one field | now clears-and-adds, mirroring `set_postings_field`/`set_doc_values_field` |
| `add_norms_field` (new) | — as a method; Java has no per-writer norms opt-in at all (every indexed non-`omitNorms` field gets a norm) | added (finding 6) |
| `build_norms_output` | `IndexingChain`'s `NormValuesWriter.flush` per field | **widened to every configured field (finding 6)** |

### Findings

4. **[CORRECTNESS -> fixed]** *`execute_merge` read the base doc-values column
   for a field whose current column is a generation.* A doc-values update
   rewrites one field's whole column into a new generation and leaves the base
   pair on disk, superseded; which generation is current is recorded on the
   field's own `FieldInfo.docValuesGen` in the segment's **newest** `.fnm`
   (`field_updates`' module comment, ported from
   `ReadersAndUpdates.writeFieldUpdates`). `execute_merge` opened
   `{seg}.dvm`/`{seg}.dvd` — the base pair, unconditionally — so merging such a
   segment would have resurrected the pre-update values into a valid,
   checksummed, `CheckIndex`-clean segment.
   It was never reachable, because c22's finding 18 kept `segment_stats` from
   offering the segment at all. It is recorded as CORRECTNESS rather than
   MISSING because the containment was the *only* thing standing between the
   code and the loss, and finding 5 removes the containment.
   Fixed by resolving each field to its current column through
   `field_updates::read_current_field_infos` and
   `field_updates::read_current_column` — the same two functions the update
   path reads *its* base from, so the merge and the update cannot disagree
   about where a column lives. Distinct `.dvm`/`.dvd` pairs are opened once
   each (a segment's base pair covers every field no update has touched, so a
   naive per-field read would copy it once per field). The merged segment folds
   every generation back into one base column (`doc_values_gen == -1`), which
   is what `IndexWriter.mergeMiddle` produces.
   Only the *location* of a column comes from the generational `.fnm`; the
   merged schema stays this writer's own `self.fields`. That is deliberate: the
   per-source `.fnm`s disagree about `doc_values_type` for a field only some
   segments have a column for, and `reconcile_field_numbers`' port of
   `FieldInfo.verifySameSchema` would reject the merge outright.
   *Test*: `a_segment_with_a_doc_values_update_merges_at_its_newest_generation`,
   which exercises **both** shapes an update takes in one segment — `soft` has
   no base column at all (the generation is the field's only column, Java's
   `FieldInfos.FieldNumbers.constructFieldInfo` case) while `payload` has only
   a base BINARY column the merge must carry forward untouched — and asserts
   each merged document's value against its stored `id`, plus
   `doc_values_gen == -1` on the merged segment and a clean `check_index`.
   **Verified to fail** with the resolution stubbed back out to the base
   column: the `debug_assert!` fires first ("every doc-values entry a source's
   .dvm declares must reach the merge"), which is that assertion doing exactly
   the job c22 added it for.

5. **[MISSING -> fixed]** *A segment with a doc-values update was never
   merged.* c22's finding 18 and second carry-over: `segment_stats` withheld
   any `doc_values_gen != -1` segment from the merge policy, "because its
   newest column lives in generational files no `.si` lists and `execute_merge`
   does not open". Both halves are now false — finding 4 opens them, and the
   gate reads `SegmentCommitInfo::files`, which lists them. The exclusion is
   dropped, and **`segment_stats` now withholds nothing**.
   That is a claim this method no longer has to make on its own. c22's verdict
   ended by observing that dropping an exclusion "moves the burden onto
   `execute_merge` to open *every* format the flush can write, and there is no
   mechanism that notices when it does not." Finding 1 is that mechanism, which
   is why this exclusion could be dropped where c22 had to keep it.
   *Test*: the rewritten
   `a_segment_carrying_only_a_doc_values_update_generation_is_never_auto_merged`,
   which is finding 4's test — the old test asserted the segment survived every
   merge round; its replacement asserts the merge happens *and* carries the
   right values.

6. **[MISSING -> fixed]** *`IndexWriter` could write norms for one field.*
   `norms_field: Option<NormsFieldConfig>` with only a `set_norms_field`, so
   finding 3's merge-side widening had nothing to exercise it: a two-norms-field
   merge was inexpressible from this writer. Real Lucene has no such limit —
   every indexed field whose `FieldInfo.omitNorms` is false gets a norm, and
   `Lucene90NormsConsumer` writes them all into one pair.
   `norms_fields: Vec<NormsFieldConfig>` with `add_norms_field` alongside
   `set_norms_field`, matching the accumulate/replace pair
   `set_postings_field`/`add_postings_field` and
   `set_doc_values_field`/`add_doc_values_field` already have.
   `build_norms_output` builds one dense column per configured field, field
   numbers ascending, into one `norms::write_fields` call;
   `fields_with_per_field_attributes`' `omit_norms` rule became a set
   membership test.
   *Tests*: `an_automatic_merge_carries_norms_for_every_field` (two fields
   whose per-document norms differ — `id` is one token, `body` is `rank/10` of
   them — so a column written twice, or one field's column read for the other,
   is visible; asserted per merged doc id against the stored `id`, with
   `check_index` clean); `add_norms_field_gates_and_deduplicates` (the three
   refusals — unknown field, unindexed field, `omit_norms` field — plus "naming
   the same field twice is a no-op", since `norms::write_fields` rejects a
   duplicate field number and an accumulating entry point that let one through
   would fail the *flush* rather than the call, plus `set_norms_field(None)`
   clearing the whole accumulated list rather than one entry).

8. **[INTENTIONAL]** *The gate refuses the merge rather than warning.* A
   `debug_assert!` would fire in every test and nowhere in release; a warning
   has nobody to read it. Refusing is strictly better than performing, because
   a merge that runs is a merge whose sources get deleted — the loss is
   permanent and unobservable. The risk c22's finding 21 named (an error out of
   `auto_merge` propagates out of `commit` and leaves an index permanently
   un-committable) does not apply: this fires only when the *code* is wrong,
   never on a legitimate segment shape, and a build in which it fires has a
   dropped format either way.

9. **[INTENTIONAL]** *`StoredFields::is_opened` is unconditionally `true`.*
   `MergeSource::reader` is not `Option`: a source without stored fields cannot
   be constructed. Asserted rather than assumed, in
   `every_optional_format_is_refused_when_the_si_has_it_and_the_source_does_not`.

10. **[INTENTIONAL]** *`.cfs`/`.cfe` are an unknown extension, not a named
    non-format one.* `IndexWriter` always flushes loose files
    (`use_compound_file: false` in `build_and_write_segment`), so a compound
    source never reaches the merge; if one ever did, its formats would not be
    visible in the `.si` at all and the gate would be answering the wrong
    question. `UnknownSegmentFormat` names it instead of a `.fdt`-not-found
    from three frames deeper. Pinned by
    `an_extension_no_format_claims_is_refused_rather_than_ignored`.

### Verdict

Swept clean. `segment_stats` withholds nothing, and what used to be a reading
obligation is now a check.

---

## crates/lucene-index/src/merge_policy.rs

Java counterpart: `index/TieredMergePolicy.java`.

**No change, and that is the finding**, for the second batch running. Nothing
about merge *completeness* belongs here: `TieredMergePolicy` decides which
segments to merge from size and deletes alone and never asks what a segment
contains (c22 finding 20 re-checked this against the pinned source). This
batch's change that reaches the merge policy is `segment_stats` handing it more
candidates (finding 5), which is a caller change.

### Verdict

Swept clean; no completeness work belongs here.

---

## Measurements

**Nothing this batch changed is on a measured path, and the measured paths were
re-run to confirm it.**

- `merge_norms` loops over a candidate list that has one entry in every
  pre-existing case, and `norms::write_single_dense_field` was already a
  one-element wrapper over `norms::write_fields` — so the single-field path is
  byte-for-byte the same calls.
- `merge_numeric_doc_values` lost two error branches and gained a `match` on an
  `Option` that was already being unwrapped.
- `execute_merge`'s doc-values opening reads each distinct `.dvm`/`.dvd` pair
  once, deduplicated by `(doc_values_gen, per-field component)`, which for a
  segment with no updates is exactly the one base pair it read before. A
  generational segment adds one small `.fnm` read.
- `check_format_coverage` is O(files x formats) per merge — roughly thirty
  string suffix comparisons against seven small tables, once, against a merge
  that reads and rewrites every file in the segment.
- `benchmarks/rust-runner/src/merge_bench.rs` drives `merge::merge_*_segments`
  with hand-built `MergeSource`s and never calls `execute_merge`, so c22's
  152x sorted stored-fields figure and c4/c8's fast paths are untouched **by
  construction**, not merely by measurement.

`benchmarks/rust-runner/src/merge-bench`, 4 segments x 20 000 documents,
`--release`, re-run in this session (the "before" arm of each row is re-run
here too, over the same inputs, exactly as c4/c8/c22 measured it):

| scenario | c22's figure | this session | verdict |
|---|---|---|---|
| stored fields, BULK (no deletions) | 520-643x | **339.8x** (364.8 ms -> 1.1 ms) | unchanged path |
| stored fields, DOC (1/3 deleted) | 23-26x | **25.8x** (245.0 ms -> 9.5 ms) | unchanged path |
| stored fields, VISITOR (renumbered fields) | 26x | **22.0x** (535.4 ms -> 24.4 ms) | unchanged path |
| postings k-way merge | 9.8-10.8x | **9.6x** (61.8 ms -> 6.5 ms) | unchanged path |
| BKD 1-D `points::write` | 2.6x | **3.1x** (144.3 ms -> 47.2 ms) | unchanged path |
| term vectors, merge BULK | 758 205x | **476 882x** (339 518.7 ms -> 0.7 ms) | unchanged path |
| term vectors, merge PER-DOC | 3 078x | **2 808x** (339 518.7 ms -> 120.9 ms) | unchanged path |
| **index-sorted merge vs the same sources concatenated** | 13.2-23.2 ms, **14.7-15.5x** | **11.3 ms vs 0.8 ms, 14.1x** | **c22's 152x fix intact** |

The last row is the one that matters and the only one immune to machine load,
because both arms merge the same sources in the same process: 14.1x is inside
c22's 14.7-15.5x band, so `copy_chunks_with_cursor` (c22 finding 9, the 152x)
is still doing its job. The absolute ratios in the other rows swing with
contention -- `c24-arith-codecs` was compiling the workspace through most of
this session -- and their "before" arms are fixed slow algorithms, which is
what makes a ratio move without the "after" moving. Every "after" figure is
within noise of c22's.


11. **[PERF, measured]** See the table above: every figure is within run-to-run
    noise of c22's, and the two the brief names specifically — c22's 152x
    sorted stored-fields merge and c4/c8's BULK/DOC/VISITOR fast paths — are
    unchanged.

---

## Verification

**The standard, set by c17 and c22**: a wrong doc map or a dropped format
produces a well-formed, checksummed, `CheckIndex`-clean segment, so every new
guarantee needs a negative control.

### The gate, on the real path

Each of the five formats a source segment can carry was stubbed out of
`execute_merge`'s opening loop, one at a time, and the suite re-run. Every one
was caught, naming the segment, the format and the exact files:

| stubbed out | what the gate said |
|---|---|
| norms (c22 finding 14's exact shape) | `MergeFormatNotOpened { segment: "_1", format: "norms", files: "_1.nvm, _1.nvd" }` |
| postings (c22 finding 23) | `... format: "postings", files: "_2_Lucene104_0.doc, _2_Lucene104_0.psm, _2_Lucene104_0.tim, _2_Lucene104_0.tip, _2_Lucene104_0.tmd, _2_Lucene104_0.pos, _2_Lucene104_0.pay" }` |
| doc values (c22 finding 22) | `... format: "doc values", files: "_2_Lucene90_0.dvm, _2_Lucene90_0.dvd, _2_Lucene90_0.dvs" }` |
| term vectors | `... format: "term vectors", files: "_2.tvd, _2.tvx, _2.tvm" }` |
| KNN vectors | `... format: "KNN vectors", files: "_2_Lucene99HnswVectorsFormat_0.vec, ....vemf, ....vex, ....vem" }` |

Three of these are c22's findings 14, 22 and 23 reproduced deliberately: the
defects that took a Tier-2 review to find are now compile-and-run failures.
(`has_blocks`, c22's finding 24, is not a format and is not in this gate's
scope — it travels on `MergeOptions`.)

The by-hand stubs are the proof that the check is wired to the real path; the
permanent equivalents are the six unit tests listed under finding 1, which
exercise `check_format_coverage` directly — including
`a_format_the_si_lists_and_the_merge_never_opened_is_refused`, which builds a
real flushed segment and a real `MergeSource` and hands the checker the file
list a norms-bearing flush actually produces.

### Finding 4's negative control

The generational resolution was stubbed back to the base column (one line:
zero every `doc_values_gen` after reading the current `.fnm`), and
`a_segment_with_a_doc_values_update_merges_at_its_newest_generation` fails —
on c22's own `debug_assert!`, "every doc-values entry a source's .dvm declares
must reach the merge", which is the first time that assertion has fired for a
real defect.

### The completeness assertion

`every_format_the_flush_writes_reappears_in_the_merged_si` is the observable
other half of the gate. It configures a writer with every format the flush path
has an opt-in for — stored fields, postings **with positions and offsets**,
term vectors, two NUMERIC doc-values columns, norms on **two** fields, KNN
vectors — flushes three unmerged sources, and asserts:

- the sources between them carry **every** `SegmentFormat` except `Points`
  ("or this test proves nothing about the ones it skipped" — a format quietly
  dropping out of the configuration fails here rather than silently narrowing
  the gate);
- the merged segment is a new one, not one of the sources;
- the merged `.si`'s format set **equals** the sources' — classified by the
  same `SegmentFormat::for_extension` the gate uses, so the test and the check
  cannot disagree about what a format is;
- `check_index` is clean.

The gate says "opened"; this says "written". Neither alone is the property.

### Coverage

`cargo llvm-cov -p lucene-index --lib --summary-only`, with its own
`CARGO_TARGET_DIR` (per c10's note that a figure taken while anything else is
building the workspace is not a figure), lines:

| file | lines | this batch |
|---|---|---|
| `index_writer.rs` | **98.44%** | c22 measured 98.25% |
| `merge.rs` | **98.54%** | c22 measured 98.50% |
| `merge_policy.rs` | **99.06%** | unchanged |
| `lucene-index` overall | **98.19%** | |

All three above the 95%-per-file bar.

**On the brief's figure.** The brief cites c23 measuring `index_writer.rs` at
85.63% with 415 of 1319 missed lines in `execute_merge` alone. That does not
describe this tree: c22 measured the same file at 98.25% and it is 98.44% now.
The likeliest explanation is that c23's figure was taken over a tree without
c22's merge tests, or under workspace contention — the exact hazard c10's note
warns about — and it is recorded here rather than quietly dropped.

**What is actually uncovered in `execute_merge`.** Forty-six uncovered regions
between the method's first line and `read_sort_keys`' last, and **every one of
them is the `Err` edge of a `?`** on a directory read or a codec `open`:
`self.dir.open("{name}.fdt")?`, `blocktree::open(...)?`,
`norms::parse_meta(...)?`, `hnsw_vectors::HnswVectorsReader::open(...)?`, the
five `.collect::<Result<_>>()?`s, and `merge::check_format_coverage(...)?`
itself. There is no uncovered *branch* — no format arm, no sort tier, no
generational path. Driving those edges means a `Directory` that fails a named
read part-way through a merge, which this crate has no fault-injecting
directory for; `check_index.rs` has that shape (c25) and `merge`/`index_writer`
do not. That is the honest description of the remaining 1.56%, and it is a
different piece of work from this batch.

### Gates

- `cargo fmt --all` — clean.
- `cargo clippy -p lucene-index --all-targets -- -D warnings` — clean, and `cargo clippy --workspace --all-targets -- -D warnings` is at **zero** (c23 got it there; c24's in-flight `lucene-codecs` burn-down had it non-zero for most of this session, and the final run is green).
- `cargo test -p lucene-index` — **682 passing, 0 failing** (656 lib + 26 integration; the lib count moved during the session as `c25-check-index-coverage` landed tests in `check_index.rs`).
- `cargo test -p lucene-search -p lucene-ffi` — green (both depend on
  `lucene-index`'s merge; `Error::TooManyNormsFields`' removal is a public API
  change).
- `python3 scripts/check-parity.py` — `check-parity: ok`. `docs/parity.md`'s `SegmentMerger`/`MergeState` row records the gate, the multi-field norms merge, and the generational doc-values merge.
- `python3 scripts/check-arith-allows.py` — `check-arith-allows: ok (8 module(s) still unaudited)`.
- `scripts/verify-write-path.sh` — **22/22 passed**, run four times across the session (the first three runs were blocked by `c24-arith-codecs`' in-flight `lucene-codecs` edits, not by this batch). This batch adds no case; c23's 22 is the baseline and it is intact.

`c24-arith-codecs` was editing `crates/lucene-codecs` throughout this session.
Every transient breakage of its in-flight work (`term_vectors.rs`,
`points.rs`, `stored_fields.rs`, `blocktree.rs`) was waited out and re-run
rather than worked around, and **no file outside this batch's own was changed
for it**.

---

## Handoffs

- **To a `lucene-codecs` batch (`c24-arith-codecs` owns the crate):** add the
  four missing sparse variants to `doc_values::DenseField` —
  `SparseBinary(i32, &[(i32, Vec<u8>)])`, `SparseSorted`,
  `SparseSortedNumeric`, `SparseSortedSet` — with the same `IndexedDISI` +
  values-body shape `SparseNumeric` (c22) and
  `write_single_sparse_numeric_field` already have, and the same
  `docsWithFieldOffset = -2` all-missing encoding
  `Lucene90DocValuesConsumer.writeValues` uses. That is the whole blocker for
  finding 7; the merge side is then the same three-line change per type that
  finding 2 was for NUMERIC (`per_source_entry[idx] = ...and_then(...)` instead
  of two `return Err`s, and `None` reader means `None` value).
  **No `lucene-codecs` change was needed for the norms carry-over** —
  `norms::write_fields` already existed (`b6-docvalues`), so c22's stated
  blocker for its finding 13 was stale and the whole item was closable in
  `lucene-index`.

---

## Carry-over

- [ ] **Sparse-across-sources for the four non-NUMERIC doc-values types**
      (finding 7). Blocked on the `DenseField` handoff above.
- [ ] **Points are still not reachable from a flush**, so `merge_points` runs
      only against hand-built `MergeSource`s. Unchanged from b10/c4/c22. The
      gate covers `Points` and will fire the day a flush writes a `.kdd`,
      which is the intended behaviour and the reason the variant exists now
      rather than later.
- [ ] **`MergeSource` cannot carry per-source `min_version`** (b10), so a
      merged segment's `.si` reports this writer's version rather than the
      oldest source's. Unchanged; not a format, so outside the gate.
- [ ] **`ConcurrentHnswMerger`** (c10 finding 35): the graph merge is still
      single-threaded, and is the merge's dominant cost for a vector-heavy
      segment.
- [ ] **The index sort is still unreachable from the FFI boundary** (c17,
      c22): `lucene-ffi` exposes no `set_index_sort`.
- [ ] **`execute_merge`'s I/O-failure edges are uncovered** (Coverage above).
      A fault-injecting `Directory` would close them for `merge.rs` and
      `index_writer.rs` at once; `check_index.rs` already has the shape.
