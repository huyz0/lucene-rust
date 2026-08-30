# c35-norms-and-sort

Two items from `LEDGER.md`'s reconciled open list, both from tier **B**
("missing Lucene behaviour a caller can reach"), and both the largest entries
left in it:

- **Item 4, norms are opt-in per field.** Java writes norms for *every*
  indexed non-`omitNorms` field; this port required a `set_norms_field` call
  per field and rewrote every other indexed field's `.fnm` entry as
  `omit_norms: true`. A caller who indexed a text field and searched it got
  length-unnormalised BM25 -- a wrong score reachable from ordinary use.
- **Item 3, `segment_info::IndexSortField` cannot represent most real sorts.**
  It modelled `(field, reverse, missing-first-or-last)` and `parse` therefore
  *rejected* anything else. Honest, but the consequence was that an index a
  real `IndexWriter` wrote with an ordinary sort could not be **opened by this
  port at all**.

Java counterparts, all read from the pinned tree `/home/tuong/work/lucene-10.5.0`
(not the working checkout, which is `main`, 4574 commits ahead):

| Rust file | Java counterpart(s) |
|---|---|
| `crates/lucene-index/src/index_writer.rs` (`norms_field_configs`, `omit_norms_field`, `build_norms_output`, `fields_with_per_field_attributes`, `set_index_sort`, `sort_pending_buffer`, `read_sort_keys`, `describe_index_sort`) | `index/IndexingChain.java` (`writeNorms`, `PerField.setInvertState`/`finish`, `maybeSortSegment`, `validateIndexSortDVType`), `index/NormValuesWriter.java`, `index/IndexWriterConfig.java` (`setIndexSort`), `index/IndexWriter.java` (`validateIndexSort`), `search/Sort.java`/`SortField.java`/`SortedNumericSortField.java`/`SortedSetSortField.java`/`BinarySortField.java` (`toString`) |
| `crates/lucene-index/src/segment_info.rs` | `codecs/lucene99/Lucene99SegmentInfoFormat.java`, `index/SortFieldProvider.java`, `search/SortField.java` (`Provider`, `serialize`, `getIndexSorter`), `search/SortedNumericSortField.java`, `search/SortedSetSortField.java`, `search/BinarySortField.java`, `index/IndexSorter.java` (`IntSorter`/`LongSorter`/`FloatSorter`/`DoubleSorter`/`StringSorter`/`BinarySorter`), `util/NumericUtils.java` |
| `crates/lucene-index/src/segment_writer.rs` | `index/Sorter.java` (`DocMap`, `sortAndLeaveUnpacked`), `index/DocumentsWriterPerThread.java` |
| `crates/lucene-index/src/merge.rs` (`merge_norms`, `describe_written_files`, `compare_heads`, `sorted_doc_order`, `MergeSortKeySpec`) | `codecs/NormsConsumer.java` (`merge`, `mergeNormsField`), `index/MultiSorter.java`, `index/SegmentMerger.java` |
| `crates/lucene-index/src/check_index.rs` (`check_index_sort`, `sort_key_values`, `doc_values_presence`) | `index/CheckIndex.java` (`testSort`, `checkSoftDeletes`) |
| `crates/lucene-ffi/src/writer.rs` (`ffi_writer_omit_norms_field`) | no Java counterpart (C ABI glue over `IndexWriter`) |

Totals: **16 findings** -- 6 CORRECTNESS (all fixed), 8 MISSING (all fixed),
1 PERF (measured), 1 INTENTIONAL. Four of them (11-14) came from the Tier-2
review; finding 11 was a defect this batch introduced, and finding 12 was
pre-existing but only became reachable once this batch made a
`SortedNumericSortField` sort expressible at all.

`scripts/verify-write-path.sh`: **22/22** (run, not assumed; unchanged count,
but four of the cases now assert materially more -- see findings 4, 9 and 14).

`scripts/docker-test.sh gate`: **ok**, workspace coverage **98.11% lines**
(98.10% before this batch), with no file below the 95% per-file bar.

`fixtures/data/` gained exactly one directory, `sorted_index_wide/`, generated
with `scripts/gen-fixtures.sh --only GenSortedIndexWide`; the only other
fixture change is the two lines `segment-ids.txt` gained for it.

---

## crates/lucene-index/src/index_writer.rs (norms)

Java counterparts: `index/IndexingChain.java`'s `writeNorms` and
`PerField.setInvertState`/`PerField.finish`, `index/NormValuesWriter.java`.

### Method correspondence

| Rust | Java | Verdict |
|---|---|---|
| `norms_field_configs` (new) | `IndexingChain.writeNorms`' loop condition, `fi.omitsNorms() == false && fi.getIndexOptions() != IndexOptions.NONE` over `state.fieldInfos` | identical |
| `omit_norms_field` (new) | `FieldType.setOmitNorms(true)` | equivalent at the writer level (this port's schema is fixed at `IndexWriter::open`, so the knob lives on the writer rather than on a per-document `FieldType`) |
| `set_norms_field` / `add_norms_field` | *nothing* -- Java has no per-writer norms opt-in | **removed (finding 1)** |
| `build_norms_output` | `NormValuesWriter.addValue`/`flush` + `Lucene90NormsConsumer.addNormsField` | **was divergent (finding 2)**; now sparse, matching `DocsWithFieldSet` |
| `fields_with_per_field_attributes`' norms branch | *nothing* -- Java's `.fnm` and its norms files are written from one fact | **removed (finding 1)**; replaced by a `debug_assert` |
| `invert_pending_fields` | `IndexingChain.processField`'s single `PerField.invert()` fan-out | unchanged; now fed the derived norms field list |

Java with **no** Rust counterpart: `NormValuesWriter`'s `PackedLongValues`
buffering and `iwBytesUsed` accounting (this port computes the whole column at
flush from the shared invert pass, so there is nothing to buffer per document),
and `Similarity.computeNorm` as a pluggable call (only BM25 exists here -- a
separately-tracked ledger item).

### Findings

1. **[MISSING -> fixed]** *Norms were opt-in, and the `.fnm` was rewritten to
   match.* Java:

   ```java
   for (FieldInfo fi : state.fieldInfos) {
     if (fi.omitsNorms() == false && fi.getIndexOptions() != IndexOptions.NONE) {
       perField.norms.finish(state.segmentInfo.maxDoc());
       perField.norms.flush(state, sortMap, normsConsumer);
     }
   }
   ```

   -- no opt-in exists anywhere; `FieldType.setOmitNorms(true)` is the only
   knob, and it is an opt-*out*. This port required `set_norms_field`/
   `add_norms_field` per field and, for every indexed field the caller had not
   named, forced `omit_norms = true` into the `.fnm`. That forcing was itself
   correct as far as it went -- promising norms a segment does not carry is
   what `DirectoryReader.open` throws `NoSuchFileException` on -- but it meant
   the `.fnm` described a different schema than the caller asked for, and
   BM25 then scored those fields against `UNNORMED_FIELD_LENGTH` instead of
   each document's own length.
   **Fixed.** `IndexWriter::norms_field_configs` is Java's loop condition;
   `omit_norms_field` is the opt-out; `set_norms_field`/`add_norms_field` and
   the `.fnm` coercion are gone. `NormsFieldConfig` is derived per flush
   rather than stored. The FFI followed:
   `ffi_writer_set_norms_field` (the opt-in) is replaced by
   `ffi_writer_omit_norms_field`.
   *Tests*: `an_indexed_field_gets_norms_with_no_opt_in_at_all` (the fix, and
   it fails against the unfixed code: `body.omit_norms` was `true` and no
   `.nvm` existed), `omit_norms_field_removes_the_column_and_says_so_in_the_fnm`,
   `the_norms_columns_are_exactly_the_indexed_non_omitting_fields`,
   `omit_norms_field_{rejects_an_unknown_field_name,rejects_a_field_with_no_index_options,is_idempotent}`,
   `commit_with_every_field_opted_out_of_norms_stays_stored_only`, and on the
   FFI side `norms_are_written_by_default_and_omit_norms_field_removes_them`.

2. **[CORRECTNESS -> fixed]** *The norm column was dense, so "no field" and
   "empty field" were the same byte.* `IndexingChain.PerField.finish(docID)`
   runs only for a document that **contains** the field, and inside it:

   ```java
   if (invertState.length == 0) {
     // the field exists in this document, but it did not have
     // any indexed tokens, so we assign a default value of zero
     normValue = 0;
   } else {
     normValue = similarity.computeNorm(invertState);
   }
   norms.addValue(docID, normValue);
   ```

   with `NormValuesWriter` tracking a `DocsWithFieldSet`, so the column is
   **sparse**: a document that does not carry the field gets *no entry*, and
   one that carries it but tokenizes to nothing gets an explicit `0`. This
   port gave both a dense `0`.
   *Consequence*: the two are indistinguishable in the `.nvd`, and the file is
   not the file Lucene writes for the same documents. It is not a scoring
   difference on the paths this port exposes (a document without the field
   cannot match a term in it), which is why it is listed after finding 1.
   **Fixed.** `build_norms_output` computes `Option<u32>` per document --
   present exactly when the document has a `FieldValue::String` for the field,
   which is the same presence test the shared invert pass uses to decide what
   to analyze -- and emits `norms::NormsField::Dense` when every document has
   one and `Sparse` otherwise, which is `Lucene90NormsConsumer`'s own
   `numDocsWithValue == maxDoc` branch.
   *Tests*: `a_document_without_the_field_gets_no_norm_and_an_empty_one_gets_zero`
   (fails against the unfixed code: doc 1 read back `Some(0)`), plus the real
   Lucene half in finding 4.

3. **[CORRECTNESS -> fixed]** *`merge_norms` refused a sparse column and a
   source that did not declare the field.* `NormsConsumer.merge` iterates the
   **merged** `FieldInfos`:

   ```java
   for (FieldInfo mergeFieldInfo : mergeState.mergeFieldInfos) {
     if (mergeFieldInfo.hasNorms()) { mergeNormsField(mergeFieldInfo, mergeState); }
   }
   ```

   and `mergeNormsField` adds a sub only for a source whose own `FieldInfo`
   says `hasNorms()`, so a source that never declared the field, and a
   document inside a source whose column has no entry for it, both contribute
   nothing and appear as a gap. This port's `merge_norms` derived its
   candidates from which sources happened to carry norms *files* and raised
   `Error::NormsFieldMissingInSource` for either gap.
   *Consequence*: latent while norms were opt-in and the opt-in wrote a dense
   column; live the moment finding 2 made columns sparse, and live anyway for
   the ordinary case of a schema that grew (segment A has the field, segment B
   does not).
   **Fixed.** `merge_norms` takes `merged_fields` and returns
   `Vec<(i32, Vec<Option<i64>>)>`; the writer picks `Dense`/`Sparse` the same
   way the flush does. `Error::NormsFieldMissingInSource` is gone.
   `describe_written_files` lost its `norms_field_numbers` parameter and its
   `omit_norms = true` coercion with it -- the merge now writes a column for
   exactly the fields whose merged `.fnm` claims one, so there is nothing to
   coerce.
   *Tests*: `a_merged_indexed_field_keeps_the_norms_claim_its_column_backs`
   (replaces `an_indexed_field_the_merge_wrote_no_norms_for_must_omit_them`,
   whose premise was the bug), and the pre-existing merge suites, three of
   which were building an impossible segment (a field with `omitNorms == false`
   but `indexOptions == NONE`, which `hasNorms()` is false for) and are now
   built as Java would allow.

4. **[MISSING -> fixed]** *No real-Lucene evidence for the norm values this
   port writes through a whole segment.* `VerifyNorms` covers the codec in
   isolation (hand-built `SegmentInfo`/`FieldInfos`); nothing read a norm back
   out of a segment this port's `IndexWriter` produced.
   **Fixed.** `write_full_segment_fixture` gained a second indexed field,
   `title`, that **nothing configures for norms** -- which is the whole point
   -- with documents that lack it entirely (`i % 7 == 0`), documents that carry
   it empty (`i % 7 == 1`), and documents with lengths 2..6.
   `fixtures/src/VerifyFullSegment.java::checkNorms` reads every document's
   norm for both `title` and `body` through `MultiDocValues.getNormValues` and
   compares it to `SmallFloat.intToByte4(tokenCount)` -- the value
   `BM25Similarity.computeNorm` would have stored -- *and* asserts that the
   documents `NumericDocValues` skips are exactly the ones that do not carry
   the field. Against the unfixed code `getNormValues("title")` returns
   `null`. Passed on the first real-Lucene run.

## crates/lucene-index/src/segment_info.rs (index-sort model)

Java counterparts: `codecs/lucene99/Lucene99SegmentInfoFormat.java`,
`index/SortFieldProvider.java`, the four providers' `readSortField`/
`serialize`, and `index/IndexSorter.java`'s six sorters.

### Method correspondence

| Rust | Java | Verdict |
|---|---|---|
| `IndexSortField` | `SortField` (as far as `SortFieldProvider` round-trips it) | **was narrow (finding 5)**; now `(field, reverse, IndexSortKind)` |
| `IndexSortKind`, `NumericSortKey`, `StringMissingValue`, `SortedNumericSelector`, `SortedSetSelector` (new) | `SortField.Type` x `missingValue`, `SortedNumericSelector.Type`, `SortedSetSelector.Type` | identical, with the type/missing-value pairing made total |
| `read_plain_sort_field` | `SortField.Provider.readSortField` | identical |
| `read_sorted_numeric_sort_field` | `SortedNumericSortField.Provider.readSortField` | identical |
| `read_sorted_set_sort_field` | `SortedSetSortField.Provider.readSortField` | identical, including "any marker but 1/2 means no missing value" (finding 6) |
| `read_binary_sort_field` | `BinarySortField.Provider.readSortField` | identical |
| `write_sort_field` | `SortFieldProvider.write` + each provider's `serialize` | **was one shape (finding 5)**; now the byte-level inverse of `read_sort_field` for every kind |
| `float_to_sortable_int` / `double_to_sortable_long` | `NumericUtils.floatToSortableInt`/`doubleToSortableLong` | **was divergent on NaN (finding 7)**; now canonicalizes as `Float.floatToIntBits` does |
| `sortable_int_to_float` / `sortable_long_to_double` (new) | `NumericUtils.sortableIntToFloat`/`sortableLongToDouble` | identical |
| `SortKeyComparator` (new) | `IndexSorter.{Int,Long,Float,Double,String}Sorter.getDocComparator` | equivalent: resolved once instead of closed over per segment |
| `java_float_compare` / `java_double_compare` | `Float.compare` / `Double.compare` | identical (not `f32::total_cmp`, which orders a negative NaN below `-Infinity`) |
| `segment_writer::sort_key_rank` | -- | replaced by `SortKeyComparator` |

Java with **no** Rust counterpart: `IndexSorter.BinarySorter` (see finding 5's
scope statement) and `ComparableProvider`/`ComparableValues`, whose role the
merge fills with a per-source key column.

### Findings

5. **[MISSING -> fixed]** *The model could not represent most real sorts, so
   `parse` refused the file.* `IndexSortField` was
   `(field, reverse, SortMissingValue{First,Last})`, and `parse` returned
   `Error::UnsupportedSortField` for a numeric sort with no missing value
   (Java's default -- `new SortField("f", Type.LONG)` -- which Java compares as
   `0`), an arbitrary numeric sentinel, or a non-`MIN` selector.
   *Consequence*: this port could not open such an index at all.
   **Fixed.** `IndexSortKind` covers all four providers, every
   `SortField.Type` that can be an index sort, both selector enums, and every
   missing-value form including "none". `NumericSortKey` pairs the type with
   its missing value in one enum, because Java's pairing is total -- two
   independent fields would make `(INT, Some(3.5))` representable and nothing
   on disk can hold it. `write` is `parse`'s byte-level inverse for all of
   them.
   Two things are still refused, and **Java refuses both**: a
   `SortField.Type` whose `getIndexSorter()` is `null`
   (`SCORE`/`DOC`/`CUSTOM`/`STRING_VAL`/`REWRITEABLE`, which
   `IndexWriterConfig.setIndexSort` rejects outright, so no `.si` real Lucene
   wrote contains one), and a `SortedNumericSortField` typed `STRING` (an
   `AssertionError` inside Java's own `serialize`).
   *Tests*: `every_numeric_type_and_missing_form_round_trips` (15 kinds x both
   directions), `the_string_sort_field_round_trips_all_three_missing_forms`,
   `every_selector_and_provider_round_trips`,
   `sorted_numeric_sorted_set_and_binary_providers_all_decode` (hand-built
   provider bytes, so the reader and writer cannot agree on a misreading),
   `a_numeric_sort_with_no_missing_value_parses_and_compares_as_zero`,
   `an_arbitrary_numeric_missing_value_round_trips`,
   `a_type_that_cannot_be_an_index_sort_is_rejected`,
   `a_sorted_numeric_sort_field_with_a_string_type_is_rejected`. Verified
   against real Lucene in both directions -- findings 8 and 9.

6. **[CORRECTNESS -> fixed]** *`read_string_missing_marker` was stricter than
   the format.* Java's `SortedSetSortField.Provider` and
   `BinarySortField.Provider` both do

   ```java
   int missingValue = in.readInt();
   if (missingValue == 1) { ... STRING_FIRST } else if (missingValue == 2) { ... STRING_LAST }
   return new ...(field, reverse, type, null);
   ```

   -- anything else falls through to "no missing value". This port returned
   `Error::UnsupportedSortField` for any other int, i.e. refused a file real
   Lucene reads. Same class of defect as the `reverse != 1` case this module
   already got right.
   **Fixed**; `read_string_missing_marker` mirrors Java exactly.
   *Test*: `an_unknown_string_missing_marker_reads_as_no_missing_value`.

7. **[CORRECTNESS -> fixed]** *`float_to_sortable_int` did not canonicalize
   NaN.* `NumericUtils.floatToSortableInt` uses `Float.floatToIntBits`, not
   `floatToRawIntBits`, so every NaN collapses to `0x7fc00000`; Rust's
   `f32::to_bits` preserves the payload. A `FLOAT` missing value of a
   signalling NaN would therefore have been written as bytes Java never
   writes, and `write` would not have been `parse`'s inverse for it.
   Unreachable before this batch (nothing could hold such a missing value),
   introduced-and-fixed within it.
   **Fixed**; both `float_to_sortable_int` and `double_to_sortable_long`
   canonicalize, and `java_float_compare`/`java_double_compare` do the same on
   the read side, which is what makes every NaN compare equal and greater than
   `+Infinity` as `Float.compare` does.
   *Tests*: `float_missing_values_round_trip_through_the_sortable_encoding`,
   `float_and_double_sorts_use_javas_compare`.

8. **[MISSING -> fixed]** *No evidence, read direction, that this port can
   open a real Lucene index with an ordinary sort.* `fixtures/data/sorted_index/`
   covers exactly the sort this port itself writes.
   **Fixed.** New generator `fixtures/src/GenSortedIndexWide.java` ->
   `fixtures/data/sorted_index_wide/` (generated with
   `scripts/gen-fixtures.sh --only GenSortedIndexWide`, which touches nothing
   else): a real `IndexWriter` index, two commits plus a `forceMerge(1)`, whose
   `Sort` has three tiers each individually unrepresentable before this batch:

   - `rank`: `LONG` descending with `setMissingValue(42L)` -- an arbitrary
     sentinel, and one that sits *inside* the data's range, so the missing
     documents land in the middle of the order rather than at either end.
   - `multi`: `SortedNumericSortField(INT, MAX)` with **no** missing value.
   - `name`: `SortField(STRING)` descending over `SortedDocValues`, compared
     by term ordinal.

   `crates/lucene-index/tests/index_sort_wide_fixtures.rs` parses the `.si`
   tier for tier, compares this port's rendering against Lucene's own
   `Sort.toString()` from the manifest, reads the documents back in Lucene's
   physical order, reproduces that order with `SortKeyComparator` from the
   manifest's columns (applying the `MAX` selector itself), and runs this
   port's `CheckIndex` -- whose `testSort` re-reads the keys out of the
   segment's own NUMERIC, SORTED_NUMERIC and SORTED columns. Against the
   unfixed code the first `parse` call fails.

9. **[MISSING -> fixed]** *No evidence, write direction, for the widened
   encoder or for a segment ordered by a widened sort.* Two additions, both in
   `scripts/verify-write-path.sh` (still 22 cases; two of them now assert
   much more):

   - `write_segment_info_fixture` gained a `_3` segment carrying **eight**
     sort fields: all four providers, every numeric type, a `STRING` tier,
     both a `MAX` and a `MIDDLE_MIN` selector, an arbitrary `INT` sentinel, a
     no-missing-value tier, and `STRING_FIRST`/`STRING_LAST`/none.
     `VerifySegmentInfo` now compares real Lucene's `Sort.toString()` on the
     `Sort` it reconstructed through `SortFieldProvider.forName(...).readSortField(...)`
     against `describe_index_sort`'s rendering, instead of a reduced
     `field:reverse:first|last` triple -- so a wrong `SortField.Type` byte, a
     wrong selector ordinal, a missing value in the wrong encoding, or a
     missing-value marker written the sorted-set way round instead of the
     `STRING` way round all show up as a different string.
   - `write_sorted_segment_fixture` and `write_sorted_merged_segment_fixture`
     now sort `rank` with an **arbitrary** sentinel (`RANK_MISSING = 0`,
     inside `rank_of`'s -20..=29 range) instead of `Long.MAX_VALUE`, so the
     documents with no `rank` interleave with the `rank == 0` ones instead of
     landing at an end. `VerifySortedSegment` derives the expected permutation
     from the same constant and runs real `CheckIndex` at
     `MIN_LEVEL_FOR_SLOW_CHECKS`, whose `testSort` rebuilds the comparators
     from the `.si`. **Negative control measured**: writing the segment with
     `RANK_MISSING = i64::MAX` while the verifier expects `0` fails 32 checks,
     starting at the stored-fields order and including
     `KnnFloatVectorQuery for docID=733 returned 788`.

## crates/lucene-index/src/{segment_writer,merge,check_index}.rs (honouring the widened model)

### Method correspondence

| Rust | Java | Verdict |
|---|---|---|
| `SortKeySpec` / `MergeSortKeySpec` | the `(SortField, per-segment values)` pair `IndexSorter` closes over | now carry an `&IndexSortField` instead of a flattened `(field, reverse, missing)` |
| `sort_permutation` / `sorted_doc_order` | `Sorter.sortAndLeaveUnpacked` / `MultiSorter.sort` | unchanged in behaviour; resolve `SortKeyComparator` once, outside the comparison loop |
| `comparators` / `merge_comparators` (new) | -- | not-in-Java: the "this sort has no single-`i64` key" guard (finding 10) |
| `IndexWriter::sort_pending_buffer`'s key extraction | `IndexingChain.maybeSortSegment` -> `IndexSorter.getDocComparator`'s doc-values read | **widened**: applies `SortedNumericSelector` to a document's repeated field values |
| `IndexWriter::read_sort_keys` | `MultiSorter.sort`'s `IndexSorter.getComparableValues` | **widened**: reads SORTED_NUMERIC with a selector as well as NUMERIC |
| `check_index::sort_key_values` | `CheckIndex.testSort`'s per-`SortField` `getDocComparator` | **widened**: NUMERIC, SORTED_NUMERIC (both selectors) and SORTED (ordinals); dispatches on the *sort kind*, as Java does through `DocValues.getNumeric`/`getSorted`, and names a kind/type mismatch instead of guessing |
| `check_index::doc_values_presence` (new) | `PendingSoftDeletes.countSoftDeletes` | split out of `sort_key_values`, which is now sort-field-shaped |
| `IndexWriter::describe_index_sort` (was private `describe_sort`) | `Sort.toString` + each `SortField.toString` | **was LONG-only**; now every kind, and pinned against Lucene's own output by two fixtures |

### Findings

10. **[MISSING -> fixed]** *Nothing said which sorts the writers can actually
    produce.* Reading a sort and producing one are different questions, and
    the widened reader made the difference matter.
    **Fixed, and stated precisely rather than silently narrowed.**
    `IndexWriter::set_index_sort` now checks the sort field's doc-values type
    against the *kind* of sort (`IndexingChain.validateIndexSortDVType`'s
    rule: NUMERIC for a numeric `SortField`, SORTED_NUMERIC for a
    `SortedNumericSortField`) and refuses the ordinal and byte kinds with a
    new `Error::UnsupportedIndexSortKind` naming why:

    - A `SortField.Type.STRING` or `SortedSetSortField` sorts by **term
      ordinal**, and this writer assigns ordinals inside
      `build_sorted_doc_values_output` *after* the buffer is permuted -- the
      key the sort needs does not exist when the sort runs.
    - A `BinarySortField` has no single-`i64` key at all
      (`IndexSorter.BinarySorter` compares raw `BytesRef`s), so
      `IndexSortField::key_comparison` returns `None` for it.

    `FLOAT`/`DOUBLE` need no special case and are supported: Lucene's own
    `FloatDocValuesField`/`DoubleDocValuesField` store
    `Float.floatToRawIntBits`/`Double.doubleToRawLongBits` in a NUMERIC
    column, so the key is those bits and `SortKeyComparator` interprets them.
    `SortedNumericSortField` with either selector is supported end to end:
    the flush reduces a document's repeated field values, and
    `read_sort_keys` reduces the merged sources' SORTED_NUMERIC column.
    `check_index` reports a `SortedSetSortField`/`BinarySortField` sort as
    **skipped**, with the reason, rather than passing it silently -- the
    index is openable and everything else about it is checked, and saying so
    is the difference between a check that passed and one that never ran.
    `comparators`/`merge_comparators` panic if a comparator-less sort ever
    reaches a flush or a merge, because sorting by "always equal" would
    produce a segment whose `.si` claims an order its bytes do not have --
    valid files, clean checksums, wrong index.
    *Tests*: `set_index_sort_refuses_the_ordinal_and_byte_sort_kinds`,
    `a_sorted_numeric_max_selector_sort_orders_the_flush_and_survives_a_merge`
    (a corpus whose per-document MIN order is the *reverse* of its MAX order,
    so a flush that took the wrong end of the column is visible; asserted
    through `check_index`, which re-derives the comparator from the written
    `.si`), `a_sort_with_no_comparator_panics_rather_than_sorting_by_nothing`,
    `a_merge_sort_with_no_comparator_panics_rather_than_concatenating`,
    `a_sort_or_soft_deletes_field_whose_values_cannot_be_read_is_reported`
    (rewritten around the SORTED_SET arm), and
    `describe_index_sort_renders_every_kind_the_way_java_does`.

## crates/lucene-index/src/index_writer.rs + check_index.rs (SORTED_NUMERIC, from the Tier-2 review)

### Findings

11. **[CORRECTNESS -> fixed]** *A `SortedNumericSortField` with a
    `FLOAT`/`DOUBLE` key read the column in the wrong encoding.* A NUMERIC
    float column holds `Float.floatToRawIntBits` (`FloatDocValuesField`); a
    SORTED_NUMERIC one holds `NumericUtils.floatToSortableInt` (`FloatField`),
    and `SortedNumericSelector.wrap` undoes it before the sorter sees a value:

    ```java
    case FLOAT:
      return new FilterNumericDocValues(view) {
        public long longValue() throws IOException {
          return NumericUtils.sortableFloatBits((int) in.longValue());
        }
      };
    ```

    This batch's first cut fed the stored value straight to
    `SortKeyKind::Float`, so `-1.0f` (stored `0xC07FFFFF`) compared as
    `-3.99999` and the whole negative half of the ordering inverted --
    reachable in both directions, since `set_index_sort` accepted the kind and
    `check_index` would have called a real-Lucene index sorted that way
    corrupt.
    **Fixed** in `IndexSortField::key_comparison`, which returns the new
    `SortKeyKind::SortableFloat`/`SortableDouble` for `SortedNumeric` float
    keys (sentinel in the same space). Introduced and fixed within this batch.
    *Test*: `a_sorted_numeric_float_sort_undoes_the_sortable_encoding`, which
    pins the two kinds against each other on the same values.

12. **[CORRECTNESS -> fixed]** *This writer's SORTED_NUMERIC columns were not
    sorted per document, and the selector was `min()`/`max()` rather than
    first/last.* `SortedNumericDocValuesWriter.finishCurrentDoc` does
    `Arrays.sort(currentValues, 0, currentUpto)`, real
    `CheckIndex.checkSortedNumericDocValues` throws `"values out of order:
    ... for doc: ..."` without it, and `SortedNumericSelector.MinValue`/
    `MaxValue` read the *first* and the *last* stored value -- so an unsorted
    column both makes the segment one real Lucene rejects and makes the
    selector pick the wrong value. This port kept the caller's order, and
    every selector reduction re-derived min/max, which masked it.
    **Fixed** in three places: `collect_sorted_numeric_values` sorts each
    document's values (Java's own place to do it),
    `IndexWriter::sort_pending_buffer` sorts before applying the selector (it
    reads the *document*, not the column, so it has to), and the three
    selector reductions read first/last. Pre-existing, not introduced here --
    but only reachable once a `SortedNumericSortField` sort became
    expressible.
    *Tests*: the ordering check itself (finding 13) with a negative control,
    `a_sorted_numeric_max_selector_sort_orders_the_flush_and_survives_a_merge`
    (whose corpus supplies `f`'s values descending and asserts the stored
    column came back ascending), and -- the one that caught the
    `sort_pending_buffer` half -- real `CheckIndex.testSort`, which rejected
    the fixture at `docID=2` until it was fixed (finding 14).

13. **[MISSING -> fixed]** *No check that a SORTED_NUMERIC column is
    ascending.* `CheckIndex.checkSortedNumericDocValues`' `"values out of
    order"` was unported, which is why finding 12 had no local detector.
    **Fixed**: `check_index`'s SORTED_NUMERIC arm walks each document's values
    and reports a descent under `doc_values.values_decode:<field>`.
    *Test*: a third arm in
    `an_index_sort_on_a_multi_valued_sorted_numeric_field_is_verified`, over a
    segment written with `[4, 3]` on one document.

14. **[MISSING -> fixed]** *No real-Lucene evidence for the
    `SortedNumericSortField` write path.* Finding 9's additions covered the
    `.si` bytes and the arbitrary sentinel, not a *flushed or merged* segment
    ordered by a selector.
    **Fixed**: `write_sorted_segment_fixture` and
    `write_sorted_merged_segment_fixture` gained a third field, `multi`
    (SORTED_NUMERIC), and a middle sort tier
    `SortedNumericSortField("multi", INT, false, MAX)` with **no** missing
    value. `rank` has only 50 distinct values over 2 000 documents, so the
    tier really breaks ties; every document carries `multi`, supplied
    **descending**, so the writer has to sort it.
    `VerifySortedSegment.checkSortedNumericTier` checks the provider, field,
    numeric type, selector, direction and the absence of a missing value off
    `LeafMetaData.sort()`, the column is compared value by value per doc id,
    and real `CheckIndex` at `MIN_LEVEL_FOR_SLOW_CHECKS` runs both `testSort`
    and `checkSortedNumericDocValues` over it. **Measured**: this is what
    caught finding 12's `sort_pending_buffer` half --
    `CheckIndexException: segment has indexSort=... but docID=2 sorts after
    docID=3`, 37 checks failed.

## PERF

15. **[PERF -- measured]** *What making norms the default costs.* Norms are a
    byte per document per normed field, and every indexed field now gets one,
    so both the time and the space matter.

    Time, `benchmarks/rust-runner`'s `index-bench`, 50 000 documents x 40
    tokens, one indexed field, release build inside the container, three runs
    each (a new `LUCENE_RUST_OMIT_NORMS` arm is the control, alongside the
    existing `LUCENE_RUST_INDEX_SORT`/`LUCENE_RUST_VECTOR_DIM` knobs):

    | arm | us/doc (3 runs) | median |
    |---|---|---|
    | norms on (the new default) | 19.87 / 19.84 / 19.85 | **19.85** |
    | `LUCENE_RUST_OMIT_NORMS=1` | 19.19 / 19.45 / 19.00 | **19.19** |

    **+0.65 us/doc, +3.4%**, for one normed field on a 40-token body -- and
    still below the ~21.5 us/doc baseline the ledger records. The cost is the
    per-field presence vector plus one pass over the shared inverted index per
    normed field; there is no second tokenization (`invert_pending_fields` has
    analyzed the union of postings/term-vector/norms fields once since c3).

    Space: one byte per document per normed field, **or zero** when every
    document's length for that field is equal -- `norms::write_fields` takes
    `Lucene90NormsConsumer`'s constant-value encoding (`bytesPerNorm == 0`)
    then. On the benchmark corpus (uniform 40-token bodies) the whole `.nvd`
    is 59 bytes per segment and the total index grows by **352 bytes over
    13.8 MB**. On the `write_full_segment_fixture` corpus (2 500 documents,
    two normed fields, one of them sparse with varying lengths) the pair is
    **6 634 bytes** (6 495 `.nvd` + 139 `.nvm`), which is the honest number for
    a real corpus: ~1 byte per document per varying-length field plus the
    `IndexedDISI` bitset for the sparse one.

    Comparison with Java: this is the same data Lucene writes, in the same
    encodings, so the space is Java's by construction. The time is not
    comparable to Java's directly (Java's `NormValuesWriter` accumulates into
    a `PackedLongValues` as each document is inverted, where this computes the
    column at flush) -- what the A/B establishes is that the default costs
    3.4% of a commit, not that it is cheaper or dearer than Java's.

## INTENTIONAL

16. **[INTENTIONAL]** *`IndexSortField` is `PartialEq` but not `Eq`.* A
    `FLOAT`/`DOUBLE` missing value is a float and `NaN != NaN`. Nothing keys a
    map on a sort field, and the one comparison that matters --
    `IndexWriter::set_index_sort`'s congruence check against an existing
    segment's sort -- is a `==` between two sort descriptors, which
    `PartialEq` gives. The one behaviour this differs from Java in is a NaN
    `FLOAT`/`DOUBLE` sentinel, which `PartialEq` makes unequal to itself where
    Java's `SortField.equals` (boxed `Float.equals`) makes it equal. That is
    unreachable: `set_index_sort` refuses no sort over it that it would
    otherwise accept except a NaN-sentinel one being re-declared identically,
    and nothing produces such a sort. Recorded rather than worked around,
    because the alternative -- storing the sortable bits instead of the value
    -- would make the public model harder to use for every caller in order to
    serve a case none of them has.

---

## Not this batch

The working tree carried uncommitted changes from earlier batches when this
one started, so `git diff` shows more than the above. In particular the
term-vector field-**name** ordering fix in
`IndexWriter::build_term_vectors_output` (`CheckIndex.checkFields`' sorted-field
requirement, its comment attributing it to `b7`) is pre-existing work, not
c35's, and is not claimed here.

---

## Verdict

- `crates/lucene-index/src/index_writer.rs` -- **swept clean** for norms and
  for the index-sort write gate. Two ledger items closed.
- `crates/lucene-index/src/segment_info.rs` -- **swept clean**: the model is
  now everything `SortFieldProvider` round-trips, and `write` is `parse`'s
  inverse for all of it, proven both directions against real Lucene.
- `crates/lucene-index/src/merge.rs` -- norms half swept clean (sparse,
  field-infos-driven, Java's `NormsConsumer.merge`); sort half honours the
  widened model for every kind the writer can produce, including
  `SortedNumericSelector` over a merged SORTED_NUMERIC column.
- `crates/lucene-index/src/check_index.rs` -- `testSort` verifies NUMERIC,
  SORTED_NUMERIC (both selectors) and SORTED (ordinals), and
  `checkSortedNumericDocValues`' ordering check is ported. **Open**: a
  `SortedSetSortField` sort is reported as skipped because reducing a
  SORTED_SET column by a `SortedSetSelector` needs an ordinal reader
  `lucene-codecs`' `doc_values` does not expose (`sorted_set_entry` exists;
  a public per-document ordinal accessor does not). A `BinarySortField` sort
  is skipped for a structural reason, not a missing accessor. Both are
  reader-side verification gaps over indexes this port can now *open* and read
  in order; neither is producible by this writer.
- `crates/lucene-ffi/src/writer.rs` -- `ffi_writer_set_norms_field` replaced
  by `ffi_writer_omit_norms_field`. **ABI change**, deliberate: the removed
  call was an opt-in Lucene does not have.
