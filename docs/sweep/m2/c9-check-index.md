# c9-check-index

Follow-up batch closing b11's carry-over on the segment verifier: b11's
findings 25 (norms values never read), 26 (positions/offsets/payloads never
walked, term vectors never cross-checked against the inverted index), and the
vectors/HNSW gap b11 recorded as "no write path exists to check".

Files: `crates/lucene-index/src/check_index.rs`,
`crates/lucene-index/src/checksum_verify.rs`.

Java counterparts:
- `lucene/core/src/java/org/apache/lucene/index/CheckIndex.java` (Lucene
  10.5.0 at `/home/tuong/work/lucene`) — every `test*` method, `checkFields`,
  `checkTermsIntersect`, `checkDocIDRuns`, `checkImpacts`,
  `VerifyPointsVisitor`, `checkSoftDeletes`, `testHnswGraph`,
  `getConnectedNodesOnLevel`
- `lucene/core/src/java/org/apache/lucene/index/FieldInfo.java`
  (`checkConsistency`)
- `lucene/core/src/java/org/apache/lucene/index/PendingSoftDeletes.java`
  (`countSoftDeletes`)
- `lucene/core/src/java/org/apache/lucene/codecs/CodecUtil.java`
  (`checksumEntireFile`)
- `lucene/core/src/java/org/apache/lucene/util/hnsw/HnswGraph.java`

Totals: **19 findings** — 1 CORRECTNESS (fixed), 12 MISSING (11 fixed, 1
recorded as unreachable-by-construction), 3 PERF (2 fixed/bounded with
measurements, 1 reasoned), 3 INTENTIONAL. Plus one **cross-batch** finding in
`lucene-codecs/src/postings.rs` (c8's file), reported not touched.

A `quality-reviewer` pass over the finished diff found three gating issues and
thirteen advisories; all are resolved below and folded into the findings
(the gating three: a false-positive freq comparison on a `DOCS`-only
term-vector'd field, a linear `position()` that left the term-vector
cross-check quadratic despite finding 18's memo, and a leftover diagnostic
scratch test). Four checks that turned out to be unfalsifiable were removed or
reframed rather than shipped — see findings 9 and 15.

Gate: `cargo fmt --all`, `cargo clippy -p lucene-index --all-targets -D
warnings`, `cargo test -p lucene-index` — 508 lib tests + 16 integration
tests, all green.

**Vectors, explicitly**: b11's premise is out of date at the codec level and
that is the level that matters here. `check_index` opens files with the
`lucene-codecs` decoders directly (it deliberately does not build on
`lucene-search`'s reader — see the module's own doc comment), and c5 landed
real `Lucene99FlatVectorsFormat` `.vec`/`.vemf` plus
`Lucene99HnswVectorsFormat` `.vem`/`.vex` in both directions. There is also a
real Java-written `fixtures/data/vectors_index` (4 000 documents, five vector
fields, 7 911 vectors) to check against. So `testVectors` and
`testHnswGraphs` **are** expressible and are implemented here. What is still
true is the narrower statement that *this port's `IndexWriter` cannot add a
vector field*, so today's checker guards Lucene-written and
`lucene-codecs`-written vector segments rather than `lucene-index`-written
ones; when the `IndexWriter` grows vector fields, the checks are already in
place.

---

## crates/lucene-index/src/check_index.rs

### Java `CheckIndex` method-by-method correspondence

Every check in each Java method, and whether this port performs it. "b11"
means the check already existed before this batch.

#### `testLiveDocs`

| Java check | Here |
|---|---|
| `hasDeletions && liveDocs == null` → error | `liv.open` (b11) |
| `bitsCardinality(liveDocs) != numDocs` | `liv.cardinality_matches_del_count` (b11) |
| `hasDeletions == false` yet a bit is clear | there is no `.liv` file to read in that case; `commit.del_count_zero_without_del_gen` (b11) is the equivalent, and `liv.max_doc_matches_si` (b11) additionally derives `maxDoc` from the `.liv`'s own byte length rather than trusting `.si` |

#### `testFieldInfos`

| Java check | Here |
|---|---|
| `FieldInfo.checkConsistency()` for every field | performed inside `field_infos::parse`, which this module calls as `fnm.open`; `lucene-codecs`' `FieldInfo::check_consistency` is a direct port of Java's (payloads without positions, term vectors / payloads / norms on a non-indexed field, point dimension/byte-count agreement, doc-values-gen without doc values). `fnm.open` therefore *is* `testFieldInfos`. Additionally `fnm.*_vs_files` (b11) cross-checks each flag against which files the segment actually has, which Java does not do at all. |

#### `testFieldNorms` — **finding 1, was entirely missing**

| Java check | Here |
|---|---|
| `normsReader.getNorms(info) != null` for every `hasNorms()` field | `norms.entry_present:<f>` (new) |
| `checkNumericDocValues`: iterate every norm value | `norms.values_decode:<f>` (new) |
| the iterator's doc count | `norms.values_decode:<f>` re-derives it and compares against `.nvm`'s `numDocsWithField` (new) |
| `checkBulkFetchNumericDocValues` (bulk vs one-at-a-time API agreement) | not applicable: this port has one norms accessor, not two, so there is no second implementation to disagree with. INTENTIONAL, finding 17. |

#### `checkFields` (via `testPostings`) — **findings 2–8**

| Java check | Here |
|---|---|
| fields out of order | not applicable: `iter_fields()` yields a `BTreeMap`-ordered view, so the order is imposed by this port rather than read off disk. INTENTIONAL, finding 17. |
| `fieldInfos.fieldInfo(field) == null`, `indexOptions == NONE` | `postings.field_in_fnm:<f>` (new) |
| `terms.getDocCount() > maxDoc` | `postings.term_dict_shape:<f>` (new) |
| `minTerm`/`maxTerm` both-or-neither | not expressible: this port stores both as (possibly empty) byte strings, never nullables, so "absent" and "the empty term" are the same value. `postings.field_summary` checks the strictly stronger property instead — `.tmd`'s minTerm/maxTerm must *equal* the first and last enumerated terms. |
| `hasFreqs`/`hasPositions` vs `IndexOptions` | vacuous here **and in Java**: `FieldReader.hasFreqs()` *is* `fieldInfo.getIndexOptions().compareTo(...) >= 0`, and this port's `blocktree::open` takes the `FieldInfos` as its argument. `x == x`. Finding 17. |
| `hasPayloads`/`hasOffsets` vs `FieldInfo` | **not** vacuous — `.pay` exists on disk exactly when the field indexes offsets or payloads, so there is an independent witness: `postings.term_dict_shape:<f>` (new) |
| terms out of order | `postings.terms_sorted:<f>` (b11) |
| term outside `[minTerm, maxTerm]` | `postings.term_stats:<f>` (new) |
| `docFreq <= 0` | `postings.doc_freq_positive:<f>` (b11) |
| `freq <= 0` per doc | `postings.doc_ids_valid:<f>` (new) |
| `!hasFreqs` ⇒ freq reads as 1 | `postings.doc_ids_valid:<f>` (new) |
| `doc <= lastDoc`, `doc >= maxDoc` | `postings.doc_ids_valid:<f>` (b11) |
| `pos < 0`, `pos > IndexWriter.MAX_POSITION`, `pos < lastPos` | `postings.positions_valid:<f>` (new) |
| `startOffset < 0`, `startOffset < lastStartOffset`, `endOffset < 0`, `endOffset < startOffset` | `postings.offsets_valid:<f>` (new) |
| `payload.length < 1` | unreachable by construction, finding 9 |
| `docCount != docFreq` per term | unreachable by construction (b11's note, restated in finding 9) |
| `totalTermFreq != recomputed` | `postings.total_term_freq:<f>` (b11) |
| `totalTermFreq <= 0`, `< docFreq`, `!hasFreqs && ttf != df` | `postings.term_stats:<f>` (new) |
| `docFreq > terms.getDocCount()` | subsumed by `postings.field_summary`'s exact `docCount` re-derivation (b11) |
| `ord()` vs expected ordinal | not applicable: this port's `TermsEnum` has no `ord()`/`seekExact(ord)` (`FieldTerms` is trie-addressed, no ordinal index). INTENTIONAL, finding 17. |
| `nextPostings` bulk-buffer agreement | no bulk postings API in this port. Finding 17. |
| "Test skipping": seven `advance(maxDoc*i/8)` probes | `postings.advance_agrees:<f>` (new), and **stronger** than Java's: rather than only checking `advance(target) >= target`, it requires the skip-list-driven `LazyDocsCursor::advance` to land on exactly the document a linear scan of the fully decoded doc list would |
| `checkDocIDRuns` | no `docIDRunEnd` in this port. Finding 17. |
| `checkImpacts` / `ImpactsEnum` vs `PostingsEnum` | no separate impacts enum — impacts come off the same decoded block, so there is nothing to disagree. Finding 17. |
| `sumDocFreq`, `sumTotalTermFreq`, `!hasFreqs ⇒ stf == sdf`, `getDocCount()`, `size()` vs recomputed | `postings.field_summary:<f>` (b11 for the first three and `docCount`/`numTerms`; the `!hasFreqs` case is new) |
| "Cross-check terms with norms" (norm ≠ 0 ⇔ doc has terms, live docs only, plus the two counts) | `norms.agree_with_postings:<f>` (new) |
| "seek to last term" | `postings.seek_agrees:<f>` (new). Its accompanying `docFreq` recount is deliberately **not** ported — see finding 9. |
| "Test seeking by ord" (10 000 sampled ordinals, then re-seek by term) | `postings.seek_agrees:<f>` (new), re-expressed against the API this port has: up to 10 000 evenly-spaced terms are collected during the forward scan and each is re-found with `try_seek_exact` *and* `try_seek_ceil`, which must report `Found` and land on that exact term |
| `checkTermsIntersect` (4 automaton/start-term combinations) | `postings.intersect_agrees:<f>` (new): `regexp_intersect` against `.*[a-e].*` and `intersect` against the glob `*e*`, each compared term-for-term against a linear scan filtered by the same matcher |
| `fields.size() != computedFieldCount` | not applicable: `iter_fields()` *is* the field map; there is no second count. Finding 17. |

#### `testStoredFields`

| Java check | Here |
|---|---|
| decode every document, deleted included | `stored_fields.every_doc_decodes` (b11) |
| `docCount != numDocs` | `stored_fields.doc_count_matches_si` (b11) |

#### `testDocValues` — **findings 10, 11**

| Java check | Here |
|---|---|
| every field's every per-doc value decodes | `doc_values.values_decode:<f>` (b11) |
| `numDocsWithField` vs the iterator | `doc_values.values_decode:<f>` (b11) |
| BINARY value length vs `minLength`/`maxLength` | `doc_values.values_decode:<f>` (b11) |
| SORTED / SORTED_SET ordinal in `[0, valueCount)` | `doc_values.values_decode:<f>` (b11) |
| SORTED_SET ordinals strictly increasing within a doc | `doc_values.values_decode:<f>` (b11) |
| `maxOrd` actually reached, and no holes in the ordinal space | `doc_values.ords_dense:<f>` (new, finding 10) |
| `lookupOrd(i)` strictly increasing, `valueCount` terms | `doc_values.terms_sorted:<f>` (new, finding 10) |
| `checkDVIterator` (`docID()` starts at -1; `nextDoc`/`advanceExact` agree) | not applicable: this port's doc-values accessors are `value(data, entry, doc)` random-access functions, not a stateful `DocIdSetIterator` with two entry points. There is no second iterator to disagree with. INTENTIONAL, finding 17. |
| `checkDocValueSkipper` (`.dvs`) | **recorded, finding 11** |

#### `testPoints` — **finding 12**

| Java check | Here |
|---|---|
| `getPointsReader() == null` while fields have points | `points.open` (b11) |
| every packed value within the field's global min/max | `points.value_within_field_bounds:<f>` (b11) |
| every leaf's own box a subset of the field box | `points.leaf_bounds_subset_of_field:<f>` (b11) |
| `getPointCountSeen() != size()` | `points.point_count_matches:<f>` (b11) |
| `docCount > size`, `docCount > maxDoc`, `getDocCountSeen() != docCount`, doc ids in range | `points.doc_count_matches:<f>` (new, finding 12) |
| `estimatePointCount` for the three constant relations | no `estimatePointCount` in this port's `PointsReader` (its `intersect` prunes but does not estimate). Finding 17. |

#### `testTermVectors` — **findings 13, 14**

| Java check | Here |
|---|---|
| decode every doc's vectors, deleted included | `term_vectors.every_doc_decodes` (b11) |
| `fieldInfo.hasTermVectors() == false` while the doc has vectors for it | `term_vectors.fields_marked_in_fnm` (b11) |
| doc count vs `.si` | `term_vectors.doc_count_matches_si` (b11) |
| `checkFields(tfv, …, isVectors = true)` on the one-document `Fields` | `term_vectors.self_consistent` (new, finding 13): terms strictly increasing, `freq > 0` and equal to the position count, positions non-decreasing and non-negative, `endOffset >= startOffset`, start/end offset arrays present together and sized `freq`, and the per-field `hasPositions`/`hasOffsets`/`hasPayloads` header flags agreeing with what the terms actually carry |
| slow level: vector field exists in postings | `term_vectors.match_postings` (new, finding 14) |
| slow level: vector term exists in postings (`seekExact`/`prepareSeekExact`) | `term_vectors.match_postings` (new) |
| slow level: `postingsDocs.advance(j) == j` | `term_vectors.match_postings` (new) |
| slow level: vector freq == postings freq, **guarded by `postingsHasFreq`** | `term_vectors.match_postings` (new), with the same guard — a field may store term vectors while its postings omit frequencies (`IndexOptions.DOCS`), in which case the postings decoder synthesizes freq 1 while the vector carries the real one |
| slow level: positions / start offsets / end offsets / payloads equal | `term_vectors.match_postings` (new) |

#### `testVectors` — **finding 15**

| Java check | Here |
|---|---|
| `dimension <= 0` | ~~`vectors.dimension_positive:<f>`~~ -- **withdrawn by c19**: the guard is unfalsifiable in Java as well as here (`hasVectorValues()` *is* `vectorDimension > 0`, and `FieldInfo`'s constructor rejects a negative one; this port has both halves), so it could only ever pass. Finding 9's own rule, applied to a check finding 9 missed. |
| unexpected `VectorEncoding` | `vectors.field_entry_matches_fnm:<f>` (new) — and the encoding/similarity/dimension in `.vemf` are cross-checked against `.fnm`'s, which Java does not do (its `FieldEntry` reads them from one place) |
| every ordinal's `vectorValue(ord).length != getVectorDimension()` | `vectors.values_decode:<f>` (new) |
| `count != values.size()` | `vectors.values_decode:<f>` (new) |
| `ordToDoc` well-formed | `vectors.ord_to_doc:<f>` (new): every ordinal's doc in `[0, maxDoc)` and strictly increasing. Java gets this from `DocIdSetIterator`'s contract; here the sparse mapping is decoded, so it is worth asserting. |
| "search the first 64 vectors to exercise the graph" | not ported. This is a smoke test of `KnnVectorsReader.search`, not an index invariant: it asserts only that a top-10 search returns *something*. The structural properties it indirectly probes are checked directly by the `hnsw.*` family below. INTENTIONAL, finding 17. |
| `FLOAT16` encoding | no `Float16` in this port (c5 records it as out of scope). Finding 17. |

#### `testHnswGraphs` / `testHnswGraph` — **finding 15**

| Java check | Here |
|---|---|
| node ordinal outside `[0, size-1]` | not shipped: `read_field_entry` already rejects an out-of-range level-node ordinal while parsing `.vem`, and level 0's node set is the implicit `0..size`, so a check here could never fire. This port validates on the way in where Java validates on the way out. |
| a neighbour not on the node's own level | `hnsw.neighbors_on_level:<f>` (new) |
| neighbours out of order, or repeated | `hnsw.neighbors_sorted:<f>` (new) |
| connectedness from the entry point | `hnsw.entry_point_reachable:<f>` (new). Java only *reports* `N/M connected` and never fails, and neither do we — the ratio goes in the passing message. The one exception is the degenerate case Java would print as `1/N`: an entry point that reaches nothing but itself on a level with more than one node, i.e. a graph whose search can never return more than one document. |

#### `checkSoftDeletes`, `testSort`, top-level `checkIndex`

All ported by b11: `soft_deletes.count_matches`, `sort.docs_in_index_sort_order`,
`commit.*`. Re-verified against Java this batch; no divergence found.

### Findings

1. **[MISSING → fixed]** *`testFieldNorms` had no counterpart at all.* The
   module cross-checked that `.nvd`/`.nvm` existed when a field claimed
   norms, and never read a norm value. A corrupted norms payload therefore
   passed a clean run and then silently changed every BM25 score computed
   from it — the exact failure mode this module exists to prevent, on the one
   file whose corruption is *invisible* to a query (a wrong norm produces a
   wrong score, never an error).
   *Fixed*: `check_field_norms` reads every field's every norm out of `.nvd`
   (`norms.values_decode:<f>`), requires an entry to exist for every
   `hasNorms` field (`norms.entry_present:<f>`), re-derives the
   docs-with-a-value count and compares it against `.nvm`'s own
   `numDocsWithField`, and performs `checkFields`' terms-vs-norms
   cross-check (`norms.agree_with_postings:<f>`): a **live** doc's norm must
   be non-zero exactly when that doc has terms in the field's postings, and
   the two counts must match. Deleted docs are exempt, exactly as in Java
   ("norms may only be out of sync with terms on deleted documents").
   *Tests*: `norms_checks_actually_run_on_a_norms_fixture`;
   `a_zeroed_norm_for_a_doc_with_terms_is_caught` (zeroing one `.nvd` byte —
   Java's "has terms according to postings but its norm value is 0" case —
   which nothing else in the module notices);
   `a_norms_file_that_cannot_be_read_is_caught_by_values_decode` (every
   single-byte `.nvm` corruption, none silently accepted, at least one
   reported by a `norms.*` check).

2. **[MISSING → fixed]** *Positions, offsets and payloads were never
   decoded.* b11's finding 26. `checkFields`' whole positional block —
   `pos < 0`, `pos > IndexWriter.MAX_POSITION` (`Integer.MAX_VALUE - 128`),
   `pos < lastPos`, `startOffset < 0`, `startOffset < lastStartOffset`,
   `endOffset < 0`, `endOffset < startOffset` — had no counterpart, so a
   `.pos`/`.pay` was validated only by its footer. Fixed as
   `postings.positions_valid:<f>` and `postings.offsets_valid:<f>`, decoded
   through `TermsEnum::try_current_postings_and_positions` so the term's
   docs, freqs and positions all come out of one metadata decode.
   See finding 3 for how they are proved.

3. **[MISSING → fixed, with an explicitly-stated reachability property]**
   The positional predicates are extracted as `check_occurrences` and tested
   directly, because **byte corruption of a real `.pos`/`.pay` cannot reach
   them in this port**: position deltas are unsigned in both the packed and
   the vint encodings and `read_positions` restarts the accumulator at zero
   per document, so a decoded position is non-negative and non-decreasing by
   construction; a corrupt bits-per-value header or block length is rejected
   by the `for_util`/`SliceInput` layer first. That was established
   empirically, not assumed — an exhaustive sweep of every payload byte of
   both a real `.pos` and a hand-built one, across five XOR masks, produced
   `postings.terms_decode` / `postings.field_summary` / the file CRC and
   never a positional violation. So the predicates are proved against a
   *writer* emitting bad data (which is what Java's own message points at
   when it suggests "the FixBrokenOffsets tool in Lucene's backward-codecs
   module"), and the byte-level property that *nothing gets through* is
   pinned separately.
   *Tests*: `bad_positions_and_offsets_are_reported_by_the_predicate` (all
   six of Java's rejected shapes, plus a clean case that must stay silent);
   `hand_built_positional_postings_pass_every_check` (a 305-occurrence,
   offsets-and-payloads fixture with a full 128-value packed block);
   `no_single_byte_corruption_of_pos_or_pay_is_silently_accepted` and
   `no_corruption_of_a_real_pos_file_is_silently_accepted`.

4. **[MISSING → fixed]** *No seek-and-reseek consistency check.* Java
   re-finds up to 10 000 sampled terms after the forward scan; nothing here
   ever asked the dictionary for a term it had just enumerated. That matters
   because the forward scan and a seek take **different paths through
   different files**: `next()` walks `.tim` blocks linearly, `seekExact` /
   `seekCeil` navigate the `.tip` trie. A corrupt `.tip` leaves every b11
   check passing.
   *Fixed*: `postings.seek_agrees:<f>` collects up to 10 000 evenly-spaced
   terms during the scan (Java's own `min(10000, termCount)` cap) and
   re-finds each with `try_seek_exact` **and** `try_seek_ceil`, requiring
   `Found` and the exact term; plus Java's dedicated "seek to last term"
   case (without its `docFreq` recount, which is vacuous here — finding 9). The `try_*` forms are c1's, and are the point:
   with the non-`try` forms a corrupt block reads as "no such term", which
   this check would then report as a seek disagreement with a useless
   message instead of the decode error it is.
   *Negative control*: `corrupting_the_term_index_is_caught_by_the_seek_check`
   sweeps `blocktree_index`'s `.tip`; every corruption is caught, and the
   seek check specifically catches a large fraction of them.

5. **[MISSING → fixed]** *No `checkTermsIntersect`.* The pruning term walkers
   (`FieldTerms::intersect` for globs, `regexp_intersect` for regexps, both
   of which c1 taught to skip whole blocks via a dead prefix) had no
   cross-check against a linear scan anywhere in `check_index`. A pruning bug
   silently drops matching terms from every wildcard and regexp query.
   *Fixed*: `postings.intersect_agrees:<f>` runs `regexp_intersect` with
   `.*[a-e].*` — Java's own `makeAnyBinary + charRange('a','e') +
   makeAnyBinary` automaton — and `intersect` with the glob `*e*`, comparing
   each term-for-term against a linear scan filtered by the same matcher.

6. **[MISSING → fixed]** *No skip-data cross-check.* Java's "Test skipping"
   block had no counterpart. This port can do better than Java here: the
   check already holds the term's fully decoded doc list, so instead of only
   asserting `advance(target) >= target` it requires the skip-list-driven
   `LazyDocsCursor::advance` to return exactly the first doc `>= target` in
   that list, at Java's same seven probe points. That makes it a genuine
   cross-check of `.doc`'s level-0/level-1 skip data against the block
   payload it indexes — two independent structures in one file.
   *Fixed*: `postings.advance_agrees:<f>`.
   *Negative control*:
   `corrupting_the_doc_skip_data_is_caught_by_the_advance_check` builds a
   1 000-document single-term segment (enough for several full 128-doc blocks
   and therefore real skip data), sweeps every `.doc` payload byte across
   three masks, requires that none is silently accepted, and requires that
   the advance check specifically catches some.

7. **[MISSING → fixed]** *Term-level statistic bounds and term-dictionary
   shape.* `totalTermFreq <= 0`, `totalTermFreq < docFreq`, `!hasFreqs ⇒
   totalTermFreq == docFreq`, a term outside `[minTerm, maxTerm]`,
   `docCount > maxDoc`, a field in the dictionary that `.fnm` does not have
   (or has with `indexOptions=None`), and `.pay`/`.pos` presence versus the
   field's declared offsets/payloads/positions. Added as
   `postings.term_stats:<f>`, `postings.term_dict_shape:<f>` and
   `postings.field_in_fnm:<f>`; `postings.field_summary` also gained Java's
   `!hasFreqs ⇒ sumTotalTermFreq == sumDocFreq`.

8. **[MISSING → fixed]** *A corrupt block during the term scan had no name.*
   The scan now uses `try_next` (c1) and reports a decode failure as
   `postings.terms_decode:<f>` rather than reading as end-of-terms — which
   would have shown up only indirectly, as a `field_summary` term-count
   mismatch.

9. **[MISSING → recorded: unreachable by construction]** Three of
   `checkFields`' guards cannot fire in this port's representation, and
   shipping a check that can never fire is worse than not shipping it, so
   they are documented in `check_postings`' doc comment instead:
   *`payload.length < 1`* (this port spells "no payload here" as an empty
   `Vec<u8>`, so a zero-length payload and no payload are the same value);
   *a payload on a field whose `.fnm` says `storePayloads = false`*
   (`read_positions` is told whether to read payloads by that very flag —
   `x == x`); and *`freq` versus the number of decoded positions*
   (`read_positions` chops the flat stream *using* `freqs`, so the group
   length is `freq` or the decode fails). Two more, found by the review pass:
   the `docFreq` recount Java performs on the last term (b11's reason —
   `postings()` is parameterized by the claimed `docFreq`, so `docs.len()`
   always equals it), and Java's `(minTerm == null) != (maxTerm == null)`
   asymmetry (this port has no nullable there; `postings.field_summary`
   checks the stronger equality against the first and last enumerated terms
   instead). Payloads **are** cross-checked in
   the one place two independent copies of them exist on disk —
   `term_vectors.match_postings`, finding 14. An earlier revision of this
   batch shipped a `postings.payloads_valid` check; it was removed once it
   was shown to be vacuous.

10. **[MISSING → fixed]** *The SORTED/SORTED_SET ordinal space was never
    validated as a space.* b11 bounds-checked each ordinal individually.
    Java additionally requires that `maxOrd` is actually reached and that the
    ordinal space has **no holes** (an ordinal no document uses is a
    dictionary term nothing can ever match, and `valueCount` is what every
    ordinal comparator and facet counter sizes its arrays from), and walks
    `lookupOrd(0..maxOrd)` requiring the terms to be **strictly increasing**
    — the property that makes ordinal comparison equivalent to term
    comparison, which every `SortedDocValues` range query and every index
    sort on a SORTED field silently depends on.
    *Fixed*: `doc_values.ords_dense:<f>` and `doc_values.terms_sorted:<f>`
    (the latter also requiring the dictionary to hold exactly `valueCount`
    terms). *Negative control*:
    `corrupted_sorted_dv_ordinal_space_is_caught` sweeps
    `sorted_dv_index`'s `.dvd`.

11. **[MISSING → recorded]** *`checkDocValueSkipper` (`.dvs`) is not ported.*
    Java validates the skip index's global value range, per-level doc ranges
    (`minDocID(level) <= maxDocID(level)`), per-level value ranges nested
    inside the global one, and `maxValueCount`. `fixtures/data/doc_values_skip_index`
    exists and `lucene-codecs` parses `.dvs`, so this is tractable; it is
    recorded rather than rushed because the skipper's read API is shaped
    around `advance`-style iteration that this batch would have had to add,
    and the batch's blast radius is already large. The `.dvs` file is still
    opened and fully CRC-verified by `file:*` (finding 16), so corruption in
    it is detected — it is the *semantic* invariants that are missing.

12. **[MISSING → fixed]** *`testPoints` never checked `docCount`.* b11
    ported the point-count half of `VerifyPointsVisitor` and not the
    doc-count half: `docCount <= size`, `docCount <= maxDoc`, every visited
    doc id in range, and `getDocCountSeen() == docCount`. `docCount` is what
    a `PointRangeQuery`'s cost estimate and `IndexSearcher`'s pruning read,
    so a wrong one changes query plans without ever producing a wrong
    answer — invisible to every other check.
    *Fixed*: `points.doc_count_matches:<f>`. *Negative control*:
    `fewer_distinct_docs_than_declared_fails_doc_count_check` writes a `.kdd`
    whose points carry the same packed values and the same point count but
    fewer distinct doc ids, so *only* the new check can catch it (asserted:
    `points.point_count_matches` must still pass).

13. **[MISSING → fixed]** *A term vector was never checked against itself.*
    Java runs the whole of `checkFields` over each document's vectors with
    `isVectors = true`. This port decoded them and looked at nothing.
    *Fixed*: `term_vectors.self_consistent`.

14. **[MISSING → fixed]** *A term vector was never checked against the
    inverted index.* b11's finding 26, second half. This is the only place
    in a Lucene index where the same `(term, doc, freq, positions, offsets,
    payloads)` tuple is stored **twice, independently**, and nothing
    compared them — so a term-vectors writer bug produced highlights and
    "more like this" results that disagreed with search, with every checksum
    valid.
    *Fixed*: `term_vectors.match_postings` — the vector's field must exist in
    the postings, every vector term must exist there (via `try_seek_ceil`,
    c1), that term's postings must contain this document (found by
    `binary_search` over the strictly-increasing doc list, which is Java's
    `postingsDocs.advance(j)` at the same cost — see finding 18), and the
    freq, positions, start/end offsets and payloads must be equal. The freq
    comparison carries Java's `postingsHasFreq` guard: a field may legally
    store term vectors while its postings omit frequencies
    (`IndexOptions.DOCS`), and in that case the postings decoder synthesizes
    freq 1 for every document while the vector carries the real one, so an
    unguarded comparison would reject a *healthy* segment. None of the
    fixtures has a `DOCS`-only term-vector'd field, so this was found by
    review rather than by a test — recorded here because "a verifier that
    rejects a valid index is worse than one that is too permissive" is the
    whole premise of this batch.
    *Negative control*:
    `a_term_vector_term_missing_from_the_postings_is_caught` sweeps
    `term_vectors_index`'s `.tim`, requiring that no corruption is silently
    accepted and that the cross-check specifically catches some.

15. **[MISSING → fixed]** *`testVectors` and `testHnswGraphs` had no
    counterpart.* b11 recorded this as unreachable ("no vector/HNSW write
    path at all"); c5 changed that at the codec level, which is the level
    `check_index` reads at. See the "Vectors, explicitly" note above.
    *Fixed*: `check_vectors` and `check_hnsw_graphs` —
    `vectors.field_entry_matches_fnm:<f>` (`vectors.dimension_positive:<f>` withdrawn by c19 -- see above),
    `vectors.values_decode:<f>`, `vectors.ord_to_doc:<f>`,
    `hnsw.neighbors_on_level:<f>`, `hnsw.neighbors_sorted:<f>`,
    `hnsw.entry_point_reachable:<f>`.

    Java's node-range guard is **not** shipped: `read_field_entry` rejects an
    out-of-range level-node ordinal while parsing `.vem`, and level 0's node
    set is the implicit `0..size`, so it could never fire. Java's
    connectedness *report* is kept as a report (Java never fails on it), with
    one failing case added: an entry point that reaches nothing but itself on
    a level with more than one node — the graph is then one no search can
    return more than a single document from.

    *Negative controls*: these needed a technique the other formats did not.
    `Lucene99FlatVectorsReader`/`Lucene99HnswVectorsReader` verify the
    **whole file's** checksum before parsing a single field entry (c5 ported
    that), so every byte flip surfaces as `vectors.open`/`hnsw.open` and
    never reaches the graph — a sweep that only asserts "something failed" is
    therefore vacuous, and an earlier revision of this batch shipped exactly
    that mistake (its assertion was satisfied by `hnsw.open`, whose name
    starts with `hnsw.`). The tests now repair the footer CRC after
    corrupting, isolating the structural check from the checksum:
    `corrupted_hnsw_neighbours_are_caught_by_the_graph_checks` asserts both
    that corruptions get *past* `hnsw.open` and that
    `hnsw.neighbors_on_level`/`neighbors_sorted` catch some;
    `corrupted_vector_ord_to_doc_mapping_is_caught` does the same for
    `.vemf`; and `corrupted_hnsw_graph_bytes_are_never_silently_accepted` +
    `corrupted_vector_data_is_caught` pin the un-repaired direction, where
    the CRC is the guard.

    Two honest notes. `vectors.values_decode` is one property, not two: the
    `count != size` clause can only fire when the per-ordinal decode already
    failed, and that decode is itself unfalsifiable by byte corruption
    because `.vemf`'s `size * dim * byteSize == vectorDataLength` guard (c5
    finding 3) rejects an inconsistent entry at open. It guards a *writer*,
    which is the regression it is there for. And a `.vemf` byte flip with a
    repaired checksum *can* be silently accepted — it is pure metadata with
    no redundancy, so some of its bytes have no second copy to disagree
    with; their guard is the checksum, which is precisely why Lucene
    full-CRCs the file at open and why `file:*` now does too.

16. **[CORRECTNESS → fixed]** *`CheckIndex`'s `test: check integrity` step
    was a footer *shape* check, not a checksum.* Real `CheckIndex` runs
    `reader.checkIntegrity()`, which bottoms out in
    `CodecUtil.checksumEntireFile` — every byte re-read, CRC-32 recomputed,
    compared. This module's `file:*` checks called
    `codec_util::retrieve_checksum`, which validates magic / algorithm id /
    checksum-field shape and **never touches the payload**. A byte flipped in
    the middle of a `.kdd`, `.fdt`, `.tim` or `.dvd` therefore passed the
    file check outright, and was caught later only if some decoder happened
    to read that byte — never, for the bytes no check decodes. This is the
    same hazard c4 fixed on the merge path (its finding 17) and there is
    already a test in `lucene-store` asserting that `retrieve_checksum` does
    not detect payload corruption.
    *Fixed*: both `check_index`'s `file:*` loop and `checksum_verify`'s
    `verify_file` now call `codec_util::check_whole_file_footer` — c4's
    `checksumEntireFile` helper — rather than `retrieve_checksum` and a
    hand-rolled seek-to-footer-then-`check_footer` sequence respectively.
    That is now one shared implementation across `merge.rs`, `check_index.rs`
    and `checksum_verify.rs`. *Test*:
    `a_payload_byte_flip_now_fails_the_file_integrity_check` flips one `.kdd`
    byte, asserts `file:_0.kdd` fails, **and** asserts
    `retrieve_checksum` still returns `Ok` on those same bytes — i.e. that
    the old check would have passed it.

17. **[INTENTIONAL]** Java checks with no counterpart because the thing they
    compare does not exist twice in this port: `checkDVIterator`,
    `checkBulkFetchNumericDocValues`/`checkBulkFetchBinaryDocValues`,
    `nextPostings` bulk-buffer agreement, `checkImpacts`, `checkDocIDRuns`,
    `TermsEnum.ord()`/`seekExact(ord)`, `fields.size()` vs the recomputed
    field count, field ordering out of a `FieldsEnum`,
    `estimatePointCount`, `hasFreqs`/`hasPositions` versus `IndexOptions`,
    `testVectors`' KNN smoke search, and the `FLOAT16` vector encoding. Each
    is either an agreement check between two APIs where this port has one, or
    a feature this port does not implement. All are named with their reason in
    the correspondence tables above and in `check_postings`' doc comment,
    rather than silently omitted.

18. **[PERF → fixed/bounded, measured]** *The term-vectors-versus-postings
    cross-check is quadratic if written the way Java writes it.* Java pulls a
    fresh `PostingsEnum` per `(document, vector term)` pair and skips to the
    document — cheap there because the enum is lazy. This port's decoders
    materialize a term's whole postings list, so the same loop would be
    O(Σ_doc Σ_term docFreq(term)) with `docFreq` as the multiplier — genuinely
    quadratic on a field where most documents contain the same terms, which
    is the normal case. *Fixed* in two places, the second of which the review
    pass caught: (a) a per-field memo keyed by `(field number, term)` so the
    *decode* happens once per distinct term rather than once per `(document,
    term)` pair, and (b) `binary_search` rather than a linear `position()`
    to locate the document in that term's doc list — with the linear scan
    still in place the memo removed only the decode and left the lookup at
    O(rank of `doc_id`), i.e. still quadratic. Together the cost is
    O(Σ docFreq + Σ_doc Σ_term log docFreq), which is the same total as
    `check_postings`' own single pass. The memo is bounded in decoded
    *elements* (documents plus position occurrences, 2^20), not entries: one
    high-`docFreq` term with positions can outweigh thousands of singletons,
    so an entry count would not have bounded the memory. The bound is a
    constant-factor guard only — past it the memo is dropped and refills,
    which never changes the result.

19. **[PERF → measured]** *Cost of everything added here.* Release build,
    mean of five full `check_directory` runs over the real fixtures:

    | fixture | size | this batch |
    |---|---|---|
    | `blocktree_index` | 8 959 docs, 6 fields, 414 terms, positions+offsets+payloads | **23.3 ms** |
    | `points_index` | 2 000 docs, 3 point fields, 5 333 points | **7.6 ms** |
    | `vectors_index` | 4 000 docs, 5 vector fields, 7 911 vectors, HNSW | **11.0 ms** |
    | `term_vectors_index` | 3 docs with vectors | **98 µs** |
    | `norms_index` | 5 docs, 2 norms fields | **83 µs** |
    | `sorted_dv_index` | SORTED doc values | **64 µs** |

    Real Java `CheckIndex -level 3` on the same `blocktree_index` reports
    0.197 s wall for the whole segment (0.058 s for `terms, freq, prox`
    alone, on 20 threads). This port does the whole thing single-threaded in
    23.3 ms.

    **Linearity**, which is the property the brief asks about — a
    single-term segment scaled from 10 000 to 160 000 documents (16x), release
    build, mean of three `check_segment` runs:

    | documents | time |
    |---|---|
    | 10 000 | 60 µs |
    | 20 000 | 95 µs |
    | 40 000 | 155 µs |
    | 80 000 | 292 µs |
    | 160 000 | 563 µs |

    16x the data for 9.4x the time; from 20 000 on (past the fixed
    per-segment cost) it is 8x the data for 5.9x the time. No check added
    here is worse than linear in its input: the per-term work is O(docFreq)
    plus O(1) trie seeks, the seek round is capped at 10 000 sampled terms
    exactly as Java's is, the two `intersect` cross-checks are one extra
    linear scan each, the norms and doc-values passes are O(maxDoc), the
    HNSW connectedness walk is O(nodes + edges) and runs on level 0 only
    (Java runs it per level because it prints per level; this reports level 0
    only), and the term-vectors cross-check is bounded by finding 18.

    Four **memory** decisions, three of them tightened after the review pass:

    - The norms pass folds the value scan and the terms-vs-norms cross-check
      into **one** walk over `0..maxDoc`, so nothing is materialized; the
      first revision built a `Vec<(i32, i64)>` over every doc with a norm.
    - `postings.intersect_agrees` compares the pruning walker and the linear
      scan **in lockstep** rather than collecting both sides. `.*[a-e].*`
      matches most natural-language terms, so the first revision held close
      to two copies of the dictionary, twice. It also now names the first
      disagreeing term instead of only the two counts.
    - `checkFields`' per-field `visitedDocs` is returned from
      `check_postings` as a local value rather than riding on the public
      `CheckStats`, so it is dropped as soon as the norms cross-check has
      used it instead of being retained for every segment for the life of
      the run.
    - The one genuinely new resident cost that stays is
      `doc_values.terms_sorted`, which decodes a SORTED field's whole
      ordinal-to-term dictionary at once where Java holds one `BytesRef` at a
      time; on a million-term dictionary that is tens of MB. Recorded rather
      than fixed — the alternative needs a streaming `lookupOrd` cursor in
      `lucene-codecs::terms_dict` that does not exist yet, and `check_index`
      already builds `FixedBitSet`s over `maxDoc` and decodes every stored
      field.

### Cross-batch finding (not fixed here — not my file)

- **`crates/lucene-codecs/src/postings.rs` (c8's file): a corrupt `.doc`
  panics instead of erroring in a debug build.**
  `read_postings`' full-block loop ends with
  `debug_assert_eq!(r.position(), header.body_end)`. When a corrupt block
  body decodes to a different length than its header declared, that is an
  `assert_eq!` failure — a panic — rather than the `Error::Corrupted` every
  other malformed-input path returns. `CheckIndex` is precisely the tool one
  runs *on* a corrupt index, and in a debug build it aborts instead of
  reporting. Reproduced by flipping bytes in a hand-built `.doc`
  (`corrupting_the_doc_skip_data_is_caught_by_the_advance_check` has to
  `catch_unwind` around it for exactly this reason, with a comment pointing
  here). The fix is to return a corruption error instead of asserting; the
  file belongs to c8.

### A rule this batch adopted

Every finding above that had to be walked back — the payload check, the
freq-versus-position-count check, the last-term `docFreq` recount, the
`minTerm`/`maxTerm` asymmetry, the HNSW node-range guard, an
`hnsw.entry_point_reachable` whose failure condition did not mean what its
name said, and a `.vex` negative control whose assertion was satisfied by
`hnsw.open` — is the same mistake: **a shipped check must be falsifiable, and
its negative control must be able to distinguish it from the checks around
it.** A cheap mechanical gate would catch the whole class: every check name
this module emits is a `format!("<family>.<check>:{field}")` literal, so an
`xtask` could require each family to appear as a *failing* name in at least
one test assertion. Recorded for whoever owns the gate list; it is outside
this batch's files.

### Verdict

Swept clean, with three items recorded rather than fixed and each stated
precisely: finding 9 (five Java guards that are unreachable by construction
in this port's representation — deliberately *not* shipped as never-firing
checks), finding 11 (`checkDocValueSkipper`, tractable, deferred with its
reason), and finding 19's memory note on `doc_values.terms_sorted`. b11's
three open items (norms values, positions/offsets/payloads plus the
term-vector cross-check, vectors/HNSW) are all closed.

---

## crates/lucene-index/src/checksum_verify.rs

Java counterpart: `CheckIndex`'s checksum-only path +
`CodecUtil.checksumEntireFile`/`checkFooter` + `SegmentInfos.files(boolean)`.

### Method correspondence

| Rust | Java | Verdict |
|---|---|---|
| `verify_directory` | `CheckIndex` fast-check loop over `SegmentInfos.files(true)` | identical (b11 fixed the file set) |
| `verify_file` | `CodecUtil.checksumEntireFile` | same result; **was a third hand-rolled copy** of the seek-to-footer-then-`check_footer` sequence — finding 16 |
| `VerifyReport::{total,failed_count,all_passed,failures}` | `CheckIndex.Status` aggregation | equivalent |

### Findings

Covered by finding 16 above: `verify_file` now calls
`codec_util::check_whole_file_footer`, the same helper `merge.rs` runs on a
merge source before a byte-copy (c4) and that `check_index`'s `file:*` checks
now use, instead of duplicating its body. The behaviour is unchanged — it was
already a full-payload CRC — so this is a de-duplication, not a fix; the
"file shorter than a footer" case is now reported by `check_footer`'s own
guard rather than by a hand-written length test. The module doc comment was
updated to name the shared helper, and the now-unused `SliceInput` import
removed.

### Verdict

Swept clean.
