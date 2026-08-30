# c13-ffi-surface

Follow-up batch closing the C-ABI surface gaps that later batches opened but
could not close, because changing the boundary is `lucene-ffi`'s call:
c11's `Occur::FILTER`, c5's vector/HNSW subsystem, c7's delete-queue APIs and
sequence numbers, b15's own four recorded deferrals, and b15's recorded
results-registry concurrency ceiling.

Files swept: `crates/lucene-ffi/src/{handle, registry, raw, segment, query,
explain, directory_reader, writer, lib}.rs`, plus two new modules
(`vectors.rs`, and the test-only `legacy_boolean_abi.rs`). Also touched:
`benchmarks/rust-runner/src/ffi_overhead.rs` (the measurement harness b15
established) and `docs/parity.md`.

## Java counterparts

**`lucene-ffi` has no Java counterpart** — it *is* the C-ABI/JNI boundary, a
layer real Lucene does not have. Per the protocol's rule 1 no Java path is
invented for the boundary machinery (`handle`/`registry`/`raw`/`error`). The
"compare against Java" step applies only where an exported function wraps a
Lucene concept, and then against the Java class the *wrapped* function was
itself ported from:

| Exported surface (new or changed here) | Wraps | Java concept behind it |
|---|---|---|
| the occur-tagged clause array at all 8 boolean entry points | `lucene_search::BooleanQuery`/`Clause` | `search/BooleanClause.Occur`, `search/BooleanQuery.Builder` |
| `ffi_open_vectors`/`ffi_close_vectors` | `lucene_codecs::vectors::FlatVectorsReader`, `hnsw_vectors::HnswVectorsReader` | `codecs/lucene99/Lucene99{Flat,Hnsw}VectorsReader` |
| `ffi_knn_float_vector_search`/`ffi_knn_byte_vector_search` | `hnsw_vectors::search` + ord→doc + live docs | `search/KnnFloatVectorQuery`, `KnnByteVectorQuery`, `AbstractKnnVectorQuery` |
| `ffi_vectors_set_live_docs` | `live_docs::parse` | `Lucene90LiveDocsFormat` |
| `ffi_open_segment`'s new `pay_name` | `postings::PayInput::open` | `Lucene104PostingsFormat`'s `.pay` |
| `ffi_writer_{add,update}_documents` | `IndexWriter::{add,update}_documents` | `IndexWriter.addDocuments`/`updateDocuments(Term, Iterable)` |
| `ffi_writer_soft_update_document` | `IndexWriter::soft_update_document` | `IndexWriter.softUpdateDocument` |
| `ffi_writer_update_{numeric,binary}_doc_value` | `IndexWriter::update_*_doc_value` | `IndexWriter.update{Numeric,Binary}DocValue` |
| `ffi_writer_delete_documents_by_query` | `IndexWriter::delete_documents_by_query` | `IndexWriter.deleteDocuments(Query...)` |
| `ffi_writer_{add_postings_field, add_term_vector_field, set_custom_freq_postings_field, set_norms_field, add_document_with_custom_freq_terms}` | the same-named `IndexWriter` methods | `IndexWriter` field configuration (this port's own shape, see b9) |
| `out_seq_no` on every mutating writer call | `SeqNo` (`i64`) | every `IndexWriter` mutator's `long` return |
| `handle::SHARDS`, `registry::Sharded` | — | **no Java counterpart** (boundary machinery) |

Totals: **24 findings** — 9 CORRECTNESS (all fixed), 12 MISSING (9 fixed, 3
recorded with named blockers: findings 5, 18 and 22), 1 PERF (fixed and A/B
measured), 2 INTENTIONAL. Findings 20-24 came out of the Tier-2
`quality-reviewer` pass on this batch's own diff, run after the gate was
green; four of them were real and are fixed, the fifth (finding 22) is this
batch's one open item.

---

## `crates/lucene-ffi/src/query.rs` (+ `explain.rs`, `directory_reader.rs`)

Java counterparts: `search/BooleanClause.java`, `search/BooleanQuery.java`,
`search/IndexSearcher.java` (`maxClauseCount`).

### Method correspondence

| Rust | Java | Verdict |
|---|---|---|
| `read_boolean_query` (new) | `BooleanQuery.Builder.add(Query, Occur)` + `build()` | ported; the wire format is this boundary's own, the semantics are Java's |
| `check_clause_count` | `BooleanQuery.Builder.add`'s `TooManyClauses` throw | identical bound (1024), now one array length instead of three |
| `clause_field_names` (new) | — (norms-map plumbing, no Java counterpart) | not-in-Java |
| `push_clause` | `clauseSets.get(occur).add(query)` | identical (Java's `Occur` declaration order) |
| ~~`read_term_clauses`~~ | — | **removed**; superseded by `read_boolean_query` |
| every `ffi_search_boolean_query*`, `ffi_explain_boolean_query` | `IndexSearcher.search(BooleanQuery, ...)` | unchanged semantics, new parameter block |

Java methods with no Rust counterpart at this boundary, still: `Builder.add`
for any `Query` that is not a `TermQuery` or a `BooleanQuery` (finding 5).

### 1. [MISSING] `Occur.FILTER` was not expressible at all

Java: `BooleanClause.Occur.FILTER` — "like `MUST` except that these clauses
do not participate in scoring". c11 landed it in `lucene-search` (a `filter:
Vec<Clause>` bucket, threaded through both lazy paths, the MAXSCORE entry and
the multi-segment stats walk) and measured it **44% cheaper than the
equivalent `MUST`**.

Us: the eight boolean entry points took three clause arrays — `must`,
`should`, `must_not` — so a JVM caller could not build the cheaper query at
all. c11 flagged this as b15's call because closing it changes the C ABI.

Consequence: every filter-shaped OpenSearch query (which is most of them:
`bool.filter` is the default for structured clauses) crossed this boundary as
a `MUST` and paid for scoring nobody reads.

Resolution: **fixed**, by replacing the parameter block rather than extending
it — see finding 2 for why. `OCCUR_FILTER = 1` is Java's own ordinal, so a
JNI caller sends `occur.ordinal()` straight through. Tests:
`a_filter_clause_matches_exactly_what_the_same_must_clause_matches` (the
matched set is *identical*, which is the half `FILTER` must not change),
`a_filter_only_query_matches_rather_than_matching_nothing` (Java's
pure-negative test is `clauses.size() == MUST_NOT count`, which a filter
clause fails — a filter-only query matches at score 0, it is not a non-match),
and `a_filter_clause_contributes_no_score_where_the_same_must_clause_does`,
which reads the scores back through `ffi_scored_results_copy` and asserts
every filtered hit scores *strictly lower* — the half that would otherwise be
untested.

### 2. [MISSING] The wire format made every new `Occur` an ABI break

Not a Java divergence — a boundary-design one, and the reason finding 1 had
been deferred rather than done. Adding a fourth `(fields, field_lens, terms,
term_lens, count)` bucket for `FILTER` would have been the second C-ABI break
for the second `Occur`, with a fifth bucket waiting for whatever came next.
The brief asked explicitly for a design where "the *next* clause kind does not
require another ABI break."

Resolution: **fixed.** One flat, `Occur`-tagged, **parent-indexed** clause
array replaces the three buckets (`clause_occurs`, `clause_kinds`,
`clause_fields`/`_lens`, `clause_terms`/`_lens`, `clause_parents`,
`clause_params`, `clause_count`, plus a `minimum_should_match` scalar). Nine
parameters where there were sixteen. What that buys:

- **An `Occur` is a value, not a signature.** A new one is a new tag.
- **A `(field, term)`-shaped leaf clause kind is a value too** —
  `PrefixQuery`, `WildcardQuery`, `RegexpQuery`, `FuzzyQuery`,
  `TermInSetQuery` are each a new `clause_kinds` tag and nothing else.
- **Nesting is free** (finding 3), via `clause_parents`.
- `clause_params` carries an `i32` attribute per clause — it is why nested
  `minimumNumberShouldMatch` (finding 4) needed no new array.

Where the format's edge actually is, recorded on `read_boolean_query` so the
next reader does not have to rediscover it: a clause kind carrying something
the format has no room for still costs a change — a `PhraseQuery`'s ordered
term *list*, or a `BoostQuery`'s / `DisjunctionMaxQuery`'s `f32`. Each needs
one more parallel array.

**This is a deliberate, one-time ABI break**, taken now while the JNI wrapper
class does not exist in this repo (it is explicitly out of scope, see
`lib.rs`'s module doc). Everything the old format could express is proved
unchanged by finding 19's bridge.

### 3. [MISSING] Nested `BooleanQuery` clause construction (b15's deferral)

Java: a `BooleanQuery.Builder` clause may be another `BooleanQuery`.
`lucene_search::Clause::Boolean` has supported this since task #25, and
`explain_clause` has always been able to explain it — but nothing could
*build* one from wire input, so `+a +(b c)` was not expressible over the C ABI
and `ffi_explain_boolean_query` could never be asked about one.

Resolution: **fixed**, as a property of finding 2's format:
`clause_parents[i]` names the enclosing `BOOLEAN` clause, `-1` for top level.
Two invariants make it safe rather than merely working:

- **`parent < i` is enforced**, so a cycle is *unrepresentable* rather than
  detected — and one reverse pass over the array therefore always sees every
  child before its parent, so the tree is built with **no recursion**.
- **`MAX_CLAUSE_DEPTH = 32`.** Not a Java limit (Java has none; only
  `maxClauseCount`). A boundary-safety limit: nested clauses are *evaluated*
  recursively (`resolve_clause_docs`) and *dropped* recursively (a `Box`
  chain), so caller-controlled nesting is caller-controlled stack depth — and
  a stack overflow is an **abort**, which `catch_unwind` cannot contain. Same
  class of defect as b15's finding 4, reached by a different route.

Tests: `a_nested_boolean_clause_is_evaluated_as_its_own_subquery` (compares
`+cat +(dog bird)` against the union of the two flat conjunctions),
`clause_order_within_a_bucket_is_preserved` (the bottom-up build appends
back-to-front; the per-bucket reverse is what restores caller order, and
nothing else would have caught its absence),
`nesting_deeper_than_the_cap_is_rejected` (and exactly at the cap still
accepted), and every malformed shape in
`every_malformed_clause_array_is_an_invalid_argument_not_a_panic`.

### 4. [MISSING] `minimumNumberShouldMatch` had no wire representation

Java: `BooleanQuery.Builder.setMinimumNumberShouldMatch(int)`. This port's
`BooleanQuery` has had the field since b12, and `search_boolean_query`
implements Java's exact gating semantics — but no FFI entry point took the
value, so every query crossing this boundary had `minimum_should_match == 0`.
Found while designing finding 2's format, not reported by any earlier batch.

Consequence: an OpenSearch `bool` query with `minimum_should_match` could not
be executed correctly through this ABI at all — it silently became "any
`should` clause matches".

Resolution: **fixed.** A `minimum_should_match: i32` scalar per entry point
for the root query, and `clause_params[i]` for a nested `BOOLEAN` clause's
own. Negative values rejected (`InvalidArgument`) rather than widened into a
`usize`. Test:
`minimum_should_match_narrows_the_result_at_the_root_and_when_nested`
asserts the narrowed set is a strict subset, and that the same three clauses
nested under one `BOOLEAN` clause with its own mSM produce exactly the
root-level answer.

### 5. [MISSING, recorded] Non-`(field, term)` clause kinds

`Clause::{Phrase, MultiPhrase, DisjunctionMax, ConstantScore, Boost, Wildcard,
Prefix, Fuzzy, Regexp, Span, PointsRange, MatchAllDocs, MatchNoDocs,
TermInSet}` are still not constructible. Five of them (`Wildcard`, `Prefix`,
`Fuzzy`, `Regexp`, `TermInSet`) are now *one tag value* away and need no ABI
change at all; the rest need an attribute the format has no room for.

**Recorded, not fixed** — this batch's job was to make the *next* kind cheap,
and it is; adding fourteen clause kinds is a different task with its own
per-kind semantics to verify. Named on `read_boolean_query` and in
`docs/parity.md`.

### 6. [CORRECTNESS] The clause cap counted three lists, and missed nested clauses

b15's finding 20 moved the `maxClauseCount` check from per-list to
per-query precisely because three separately-capped lists let a caller pass
`3 x 1024`. The occur-tagged format makes that structural: **one array, one
length**. It also closes a hole b15 could not have seen, since nested clauses
were not constructible then — the cap now counts the whole tree.

Deliberately **stricter than Java** for a nested query (Java allows 1024 per
nesting level, since each level has its own `Builder`). The cap exists here as
a denial-of-service guard on a caller-supplied count, and "1024 total" is the
bound that guard wants, not "1024 per level times however many levels the
caller asked for". Stated as such on `check_clause_count`. Test:
`the_clause_cap_is_the_whole_array_and_counts_nested_clauses_too`.

### 7. [CORRECTNESS] The scored paths' norms map missed nested and filter fields

Introduced by finding 3: `query.rs`'s two scored boolean entry points,
`explain.rs`, and `directory_reader.rs`'s four multi-segment ones each built
their per-field norms map by iterating `must`/`should`/`must_not` **at the top
level only**. With nesting and `filter` now constructible, a field mentioned
only inside a nested clause (or only in `filter`) would be absent from the
map — and an absent field silently falls back to
`lucene_search::similarity::UNNORMED_FIELD_LENGTH`, i.e. a *wrong BM25 score*
rather than an error, the same failure mode b15's finding 17 fixed for
`avgdl`.

Resolution: **fixed** with one shared `query::clause_field_names`, an
iterative (explicit-stack, per finding 3's reasoning) walk of the whole clause
tree across all four buckets, replacing three separate copies of the
top-level-only iteration. `filter` fields are deliberately included even
though a filter clause is unscored: an extra map entry costs one norms open,
a missing one silently changes scores, so the safe direction is inclusion.
Test: `clause_field_names_walks_nested_clauses`.

### Verdict

`query.rs`, `explain.rs`, `directory_reader.rs`: swept-clean (findings 1-7
fixed); finding 5 open with the blocker named.

---

## `crates/lucene-ffi/src/vectors.rs` (new)

Java counterparts: `search/{KnnFloatVectorQuery, KnnByteVectorQuery,
AbstractKnnVectorQuery, TopKnnCollector}.java`,
`codecs/lucene99/Lucene99{Flat,Hnsw}VectorsReader.java`.

### Method correspondence

| Rust | Java | Verdict |
|---|---|---|
| `ffi_open_vectors` | `Lucene99{Flat,Hnsw}VectorsReader` constructors | ported (validated at open, per this crate's convention) |
| `knn_search` + the two entry points | `AbstractKnnVectorQuery.approximateSearch` / `KnnFloatVectorQuery.approximateSearch` | ported; divergence in the deletion filter (finding 10) |
| `knn_params` | `AbstractKnnVectorQuery`'s `k >= 1` check + `KnnCollectorManager.newCollector` | ported, plus OpenSearch's `num_candidates` (finding 9) |
| `check_similarity` | — (no Java parameter; `FieldInfo` owns it) | not-in-Java, deliberate (finding 11) |
| `ord_to_doc`/`to_scored_docs` | `OrdinalTranslatedKnnCollector` | ported |
| `ffi_vectors_set_live_docs` | `LeafReader.getLiveDocs()` plumbing | ported, shared with `segment.rs` |
| `similarity_ordinal` | `Lucene94FieldInfosFormat`'s pinned ordinal list | identical |

Java methods with no Rust counterpart: `AbstractKnnVectorQuery.rewrite`'s
exact-search fallback for a restrictive `filter` (this port has no query
filter to be restrictive), and `seededSearch`.

### 8. [MISSING] KNN search did not cross the boundary at all

c5 ported `Lucene99FlatVectorsFormat` and the whole HNSW stack and verified
them arc-for-arc against real Lucene (recall@10 0.9250, matching Lucene's
exactly; 47x queries/sec over brute force), c10 made `IndexWriter` able to
*write* vector fields — and nothing crossed the FFI boundary, so OpenSearch
could use none of it.

Resolution: **fixed.** `ffi_open_vectors` + `ffi_knn_float_vector_search` /
`ffi_knn_byte_vector_search` + `ffi_vectors_set_live_docs` +
`ffi_close_vectors`. Both encodings, all four similarity functions. Hits come
back as `(doc_id, score)` through the **existing** `ScoredResultsHandle` and
`results_scored.rs` accessors — a KNN hit *is* a scored doc (Java's
`KnnFloatVectorQuery` produces a `TopDocs` like any other query), so no new
results shape was invented.

**Its own handle and registry, not `SegmentHandle`.** `ffi_open_segment`
requires `.tim`/`.tip`/`.tmd`, because everything built on it needs a term
dictionary; a vector field needs none — real Lucene's `KnnVectorsReader` is a
per-segment reader entirely independent of `FieldsProducer`, and
`fixtures/data/vectors_index` is exactly such a segment (five vector fields,
no postings at all). Folding vectors into `SegmentHandle` would have made a
vectors-only segment unopenable.

**Evidence is differential against Java, not against our own expectations.**
`fixtures/data/vectors_index/manifest.properties` records, per query, the
exact `(doc, scoreBits)` list real Lucene's `KnnFloatVectorQuery`/
`KnnByteVectorQuery` returned. `knn_search_reproduces_lucene_knn_vector_query_results`
runs every one of those 60+ queries **through the exported C symbols** and
asserts the same docs in the same order with the same scores. Plus:
`without_a_graph_the_search_is_lucene_s_exact_brute_force` (the exhaustive
branch reproduces the fixture's brute-force expectations, not the graph's),
`a_field_with_no_graph_in_an_opened_vex_still_searches_exactly` (a field
below `HNSW_GRAPH_THRESHOLD` carries no graph even when `.vex` is opened),
and `a_sparse_field_returns_doc_ids_not_ordinals` (which additionally asserts
at least one returned doc id is past the field's ordinal range, so an
ordinal-for-doc-id bug could not pass).

### 9. [INTENTIONAL] `ef_search` is OpenSearch's knob, not Lucene's

Lucene's `KnnFloatVectorQuery` has no beam parameter — it searches with a
collector of exactly `k`. OpenSearch's `num_candidates` does. The brief asked
for "the `efSearch`/beam parameter", so it exists, defined so that **`0`
reproduces `KnnFloatVectorQuery` exactly** and any larger value widens the
beam and truncates back to `k`: strictly better recall for strictly more
work, never a different *kind* of answer. `visited_limit` likewise takes `0`
for Java's unlimited default. Both stated on `knn_params`. Test:
`a_wider_ef_search_never_returns_worse_hits`.

### 10. [CORRECTNESS] Deleted documents would have been returned

Java filters *inside* the graph walk: `KnnVectorValues.getAcceptOrds(acceptDocs)`
is handed to `HnswGraphSearcher` as a lazy `Bits`, so a deleted node never
enters the collector. Without an equivalent, this entry point would have
returned deleted documents as live hits — exactly b15's finding 1, reached by
a new route.

Resolution: **fixed**, but by a different mechanism, and the difference is
recorded rather than hidden. `hnsw_vectors::search` — the ported
`Lucene99HnswVectorsReader.search` — has no `acceptOrds` parameter, and adding
one is a `lucene-codecs` change (out of this batch's files). So the beam is
**widened by the segment's deleted count** and deleted docs are dropped after
the ordinal→doc translation. That is *exact*, not approximate: at most
`deleted` of the returned `beam + deleted` hits can be deleted, so at least
`k` live hits survive whenever the graph found `k` live hits at all. The cost
is a wider beam on a segment with many deletions — and once the widened beam
reaches the field's size, `hnsw_vectors::search` switches to its exact
exhaustive branch, which is what Java also does when a filter is restrictive
enough. Recorded on `to_scored_docs` with the blocker named.

`ffi_vectors_set_live_docs` and `ffi_segment_set_live_docs` now share one
`segment::decode_live_docs` (extracted, not duplicated) so the two handle
kinds cannot drift apart on `del_gen`/`del_count`/`max_doc` validation. Test:
`attached_live_docs_remove_deleted_docs_from_knn_results` asserts no deleted
doc comes back, that `k` is still filled, *and* that the surviving prefix of
the undeleted top-10 is unchanged and in order.

### 11. [INTENTIONAL] `similarity` is a cross-check, not an override

Real Lucene has no such parameter: `FieldInfo` owns the similarity and
`KnnFloatVectorQuery` uses it unconditionally. The brief asked for it, so it
exists — as a cross-check. `SIMILARITY_FROM_FIELD` (`-1`) means "the field's
own"; any other value must *equal* the field's or the call is rejected.

The reason is not stylistic: the HNSW graph's arcs encode the build-time
similarity's neighbourhood, so walking the graph under a different one
silently degrades recall with **no error at all**. A caller that believes a
field is `COSINE` and finds it is not should learn that here, not through
quietly worse results. Tests:
`a_matching_similarity_is_accepted_and_a_mismatching_one_is_not` and
`every_similarity_function_in_the_fixture_round_trips_its_own_ordinal` (all
four, both encodings).

### 12. [CORRECTNESS] `k` reached a heap allocation unvalidated

`KnnCollector::new(k, ..)` builds a `TernaryLongHeap` of `k` entries. A `k`
straight off the boundary — a negative Java `int` widened to `usize` is the
realistic way — would reach `Vec::with_capacity` and **abort**, which
`catch_unwind` cannot contain. Same defect class as b15's finding 4, in new
code.

Resolution: **fixed** in `knn_params`: `k < 1` is Java's own
`IllegalArgumentException` ("k must be at least 1"), and everything is then
clamped to the field's own vector count, which is validated against the
`.vec` file's length when the reader opens. Tests:
`k_zero_is_rejected_like_java_s_knn_vector_query` and
`an_absurd_k_is_clamped_to_the_field_size_not_allocated` (which passes
`usize::MAX` and asserts the result is the field's whole population).

### Verdict

`vectors.rs`: swept-clean (findings 8-12 resolved; 10 and 11's divergences
recorded in the code). Finding 22 (altitude) is open.

---

## `crates/lucene-ffi/src/segment.rs`

### 13. [MISSING] `.pay` (payloads/offsets) — b15's deferral, now reachable

b15 recorded `.pay` as "no consumer in this crate can use payloads yet". That
stopped being true: `lucene_search`'s `DirectoryReader` opens `.pay` per
segment (`OpenedSegments`' `pay_buf`), so the **multi-segment** entry points
had started honouring payloads while the single-segment ones structurally
could not — two entry points to the same index disagreeing about whether a
payload-carrying field is searchable at all. A phrase query on a field whose
`IndexOptions` include payloads or offsets needs it.

Resolution: **fixed.** `ffi_open_segment` takes `pay_name`/`pay_name_len`
(null = open none, which stays correct for a field carrying neither), the
file is validated with `PayInput::open` at open time so a wrong file is
`Decode` there rather than at the first query, and `query.rs`'s two phrase
entry points plus `explain.rs`'s phrase explain pass the opened `PayInput`
through instead of `None`. The boolean and term paths still pass `None`,
which stays correct (b15's finding 13: their clauses are flat `Clause::Term`,
and now also `Clause::Boolean` of them — neither needs positions). Tests
against the fixture's real Java-written `_0_Lucene104_0.pay`:
`opening_a_segment_with_its_pay_file_attaches_it` and
`a_pay_name_pointing_at_another_file_is_a_decode_error` (which also pins that
a *missing* file stays `Io`, not `Decode`).

### Verdict

`segment.rs`: swept-clean (finding 13 fixed; `decode_live_docs` extracted for
finding 10).

---

## `crates/lucene-ffi/src/writer.rs`

Java counterpart: `index/IndexWriter.java`.

### Method correspondence (the delta)

| Rust | Java | Verdict |
|---|---|---|
| `ffi_writer_add_documents` | `addDocuments(Iterable<Iterable<…>>)` | ported |
| `ffi_writer_update_documents` | `updateDocuments(Term, Iterable<Iterable<…>>)` | ported |
| `ffi_writer_soft_update_document` | `softUpdateDocument(Term, doc, Field...)` | ported, scoped to one numeric soft-delete field |
| `ffi_writer_update_numeric_doc_value` | `updateNumericDocValue(Term, String, long)` | identical |
| `ffi_writer_update_binary_doc_value` | `updateBinaryDocValue(Term, String, BytesRef)` | identical |
| `ffi_writer_delete_documents_by_query` | `deleteDocuments(Query...)` | ported over `lucene_index::DeleteQuery` (c7's closed enum) |
| `ffi_writer_add_postings_field` / `add_term_vector_field` / `set_custom_freq_postings_field` / `set_norms_field` / `add_document_with_custom_freq_terms` | the same-named `IndexWriter` methods | ported |
| `out_seq_no` on add/update/delete | every mutator's `long` return | ported |
| — | `IndexWriter.apply_merge` | **still missing** (finding 18) |

### 14. [MISSING] The sequence number never crossed the boundary (c7's A7)

Java's `IndexWriter` returns a `long` seqNo from every mutating method, and
callers use it: it is how a caller knows whether a `DirectoryReader` it holds
already reflects an operation, and how OpenSearch's translog orders
replicated operations. c7 landed the real `DocumentsWriterDeleteQueue` and
recorded (its finding A7) that this boundary dropped the number on the floor.

Resolution: **fixed.** A nullable `out_seq_no: *mut i64` on every mutating
writer entry point — an out-parameter rather than a return value because
every exported function here returns its `FfiStatus`, and null means "the
caller does not track it". An ABI change on three existing functions
(`add_document`, `update_document`, `delete_documents`), taken alongside
finding 2's. Test:
`every_mutating_call_returns_a_strictly_increasing_sequence_number` asserts
the first is `1` (Java's `DocumentsWriterDeleteQueue` starts there
deliberately, "because some APIs negate this") and that add/update/delete
strictly increase.

### 15. [MISSING] c7's four unblocked APIs were unreachable

`softUpdateDocument`, `updateDocValues`, `deleteDocuments(Query)` and block
adds all landed in c7 and none crossed the boundary.

Resolution: **fixed**, all four.

`deleteDocuments(Query...)` needed a wire format for a *recursive* query, and
gets the same kind-tagged, parent-indexed node array finding 2 established for
clauses — same `parent < i` acyclicity, same no-recursion reverse build, same
depth cap (`MAX_DELETE_QUERY_DEPTH`) and node cap
(`MAX_DELETE_QUERY_NODES = 1024`) for the same reasons. All seven
`DeleteQuery` variants (`Term`/`Prefix`/`TermRange`/`MatchAll`/`Any`/`All`/
`Not`), with a per-node flags word for a range's inclusivity and open bounds.
A new variant is a new tag value.

Tests, all end-to-end through a real writer and read back through this
crate's own read side: a prefix delete removes exactly the matching docs; a
composed `ANY(term, term)` removes both branches (proving the nesting is
real); a `[b TO *]` range honours the inclusive-lower and open-upper flag
bits; `MatchAll` empties the index (Java's LUCENE-6379 specialisation into
`deleteAll`); `softUpdateDocument` leaves **both** the old and the new
document live (the whole point of a soft delete) and refuses an empty
soft-delete field with Java's own "at least one soft delete must be present";
`addDocuments` lands its block **contiguously and in caller order** (asserted
with an order-preserving reader, because the module's existing
`read_all_live_ids` sorts and therefore cannot see ordering at all);
`updateDocuments` replaces the matched docs with the whole block; the
numeric/binary doc-values updates each have a success path and their
mirror-image type error; and every malformed node array shape is rejected.

### 16. [MISSING] Four `IndexWriter` field setters had no wrapper (b15's deferral)

`add_postings_field`, `set_custom_freq_postings_field`, `set_norms_field`,
`add_term_vector_field` (+ `add_document_with_custom_freq_terms`). Two of
these were not cosmetic: without `add_postings_field` a writer could index
exactly **one** searchable field, so a JVM caller could not build a
multi-field index at all; and without `set_norms_field` the writer produced no
`.nvm`/`.nvd` at all, so an index it built could only ever be scored with
`UNNORMED_FIELD_LENGTH` — and `ffi_open_segment`'s `nvm_name`/`nvd_name`
parameters had nothing to open.

Resolution: **fixed**, all five. `set_norms_field` has an end-to-end test that
reads the written `.nvm` back with `lucene_codecs::norms::parse_meta` and
asserts the field has a real entry — the round trip, not just the return code.
`add_document_with_custom_freq_terms` additionally enforces Java's `freq >= 1`
at the boundary, so a bad value is an `InvalidArgument` at the call that
caused it rather than a flush error many documents later.

### 17. [CORRECTNESS] Caller-misuse errors arrived as `Io`

`map_writer_error` enumerated fourteen `index_writer::Error` variants as
`InvalidArgument` and sent everything else to `Io` through a `_` arm. The list
had not grown with `lucene-index`: vector-field errors, custom-freq postings
errors, norms-field errors, the auto-flush knob validations and *all* of c7's
soft-delete/doc-values-update errors were reaching the catch-all. Every one of
them is a caller-misuse error real Lucene raises as
`IllegalArgumentException`.

Consequence: a JNI caller branching on `Io` would retry, log a disk problem,
or fail a shard for what is actually a bad argument — the exact confusion
b15's finding 6 spent a batch removing elsewhere. Found by this batch's own
new tests failing with `Io` where they expected `InvalidArgument`.

Resolution: **fixed**, and made non-recurring: **the `_` arm is gone**. Both
sides of the match are now enumerated explicitly, so the next variant added to
`lucene_index::index_writer::Error` fails to compile here and has to be
classified rather than silently becoming an `Io`. (The first cut kept the `_`
arm while claiming in a comment that it did not — caught by the Tier-2 review,
finding 21.)

### 18. [MISSING, recorded] `IndexWriter::apply_merge` — unchanged deferral

Re-assessed as the brief asked. `apply_merge` takes a `SegmentCommitInfo` the
caller must have produced by *running* `merge::merge_stored_only_segments`
itself, and this crate still exposes no way to run a merge (merging happens
only automatically, inside `commit()`, via `set_merge_policy`). Wrapping
`apply_merge` alone would be a surface no caller could drive.

**Recorded, unchanged.** Blocker named: an `ffi_merge_segments` (or
equivalent) has to exist first, which is a task of its own — it must expose
merge *selection* as well as execution, and decide what a partially-completed
merge means across the boundary.

### Verdict

`writer.rs`: swept-clean (findings 14-17 fixed); finding 18 open with the
blocker named.

---

## `crates/lucene-ffi/src/{handle,registry}.rs`

### 19. [PERF] The results registries' exclusive sections (b15's finding 11)

b15 took the registries from `Mutex` to `RwLock` and measured 6.2x on a
four-thread fan-out, then recorded the remaining ceiling: each call still
takes the **exclusive** guard twice on the results registries — once to insert
the handle the query produced, once for the caller's `ffi_close_*`.

Resolution: **fixed and measured.** The six results registries are now
`Sharded`: `SHARDS = 16` independent `RwLock<SlotMap<T>>`s each, with the
issuing shard recorded in **four bits of the handle itself**. An insert goes
to the calling thread's own sticky shard (one `AtomicUsize` round-robin
claimed once per thread, then cached in a thread-local), so N ≤ 16 threads
inserting concurrently contend on N *different* locks; a lookup or a close
reads the shard out of the handle with a shift and a mask and touches only
that one.

The four bits come from the **generation** field, not the index field
(32 → 28 bits): 2^28 reuses of one slot before the counter wraps, against a
2^24 cap on simultaneously-open handles per shard, is still an enormous
margin — and the index field, the one that must never truncate (b15's finding
5), is untouched. Handle validation is unchanged in kind: tag, shard *and*
generation must all match the slot's current occupant. The shard field is
masked, so every possible `u64` names a real shard and a garbage handle is a
lookup miss, never an out-of-range index.

Measured with **b15's own paired A/B methodology** and the same section D of
`benchmarks/rust-runner/src/ffi_overhead.rs` (400k
`ffi_search_term_query_multi_segment` calls against
`fixtures/data/blocktree_index`): the same binary built twice with **one line
changed** — `my_shard`'s round-robin modulus, `% SHARDS` against `% 1`, so
the "before" build puts every insert on one shard and is otherwise
byte-identical — then run alternately. Three rounds per fan-out, 20-core
machine at load average ~5:

| caller threads | unsharded (ns/call wall) | sharded | speedup |
|---|---|---|---|
| 1 | 1054 / 1059 / 1119 | 1048 / 1056 / 1055 | 1.00x |
| 4 | 571 / 695 | 558 / 620 | ~1.1x |
| 16 | 487 / 494 / 490 | 235 / 232 / 245 | **2.08x** |
| 32 | 494 / 480 | 222 / 226 | **2.19x** |

The single-threaded row matters as much as the others: the shard lookup costs
nothing measurable, and the decomposed "FFI boundary (C − B)" line is
unchanged too (560-589 ns vs 579-593 ns).

**The fan-out is the whole story, and it is why b15's 1.17x looked like a
ceiling.** Section D's default fan-out is 4, and four threads on a 20-core box
barely contend — which is why the same change is worth only ~10% there. At 16
and 32 threads, the fan-outs an OpenSearch node's search thread pool actually
runs at (it is sized to the core count), the single results-registry lock is
the binding constraint and removing it doubles throughput. As scaling rather
than a ratio: over its own 1-thread baseline the unsharded build reaches 2.15x
on 16 threads, the sharded one **4.46x**.

The non-results registries (`directories`/`segments`/`directory_readers`/
`writers`/`vectors`) are deliberately **not** sharded: they are written only
by open/close, once per reader lifetime rather than twice per query, so
sharding them would add a shard-selection branch to the hot *read* path to
relieve contention that is not there.

The harness gained an `FFI_OVERHEAD_THREADS` override to reach those
fan-outs; the default stays 4 so the headline number stays comparable with
b15's.

Tests: handles round-trip across threads; eight concurrent inserts demonstrably
land on more than one shard (asserted on the handles' own shard field, i.e.
the property the measurement depends on); a handle with tampered shard bits is
rejected by every other shard; `u64::MAX` is a miss, not a panic; two sharded
registries reject each other's handles; the generation wrap never bleeds into
the shard field; and the four bit-fields tile a `u64` exactly.

### Verdict

`handle.rs`, `registry.rs`: swept-clean (finding 19 fixed and measured).

---

## Tier-2 review findings (this batch's own diff)

The `quality-reviewer` subagent was run after the gate was green. It returned
four gating findings and eight advisories.

### 20. [CORRECTNESS] `docs/parity.md` contradicted itself

Row 190 (the `lucene-ffi` deferral list) was rewritten thoroughly; the three
per-module rows it supersedes were not. Row 174 still said "**No `.pay`
(payload) parameter yet**", row 182 still described the three-bucket clause
format and claimed "`live_docs` is always `None` (no `.liv` FFI surface yet)",
and row 181 still said nested-`Boolean` explanations were impossible because
those clauses could not be searched. A parity file that contradicts itself is
worse than one merely behind — invariant #7 is "updates in the same change".

Resolution: **fixed.** All three rows corrected, and a new
`lucene-ffi/src/vectors.rs` row added alongside the other per-module rows
(KNN existed only in row 190's prose).

### 21. [CORRECTNESS] Finding 17's own comment was false

The rewritten `map_writer_error` carried a comment claiming "every variant is
enumerated explicitly … so the next one added to `lucene-index` fails to
compile here" — above a match that still ended in `_ => FfiStatus::Io`. The
comment described the fix that had *not* been made. Resolution: **fixed** —
the `_` arm is gone and both sides are enumerated, which is what makes the
comment true.

### 22. [MISSING, open] KNN query policy sits in `lucene-ffi`, not `lucene-search`

`lucene-search` has no vector module at all, so Java's *query-level*
`AbstractKnnVectorQuery` policy — the `k >= 1` bound, the beam width, the
`visitedLimit` default, the ordinal→doc translation, the deletion filter —
lives in `vectors.rs` rather than one layer down. That is the wrong altitude,
and unlike every sibling entry point in this crate (`points_query.rs` wraps
`lucene_search::points_query`, `sort.rs` wraps `lucene_search::doc_value_query`).

Consequences: no non-FFI consumer can run a KNN query, so
`lucene_search::multi_segment`'s fan-out cannot — though Java's
`IndexSearcher.search(KnnFloatVectorQuery, k)` is inherently multi-leaf; and
finding 10's deletion divergence is a *search-layer* decision recorded at the
boundary, where nothing outside `lucene-ffi`'s own tests can differentially
verify it.

**Recorded, not fixed. Blocker named**: `crates/lucene-search` belonged to a
concurrently-running batch (`c12-search-features-2`) and could not be edited
by this one. The fix is mechanical — move `KnnParams`/`knn_params`/
`check_similarity`/`ord_to_doc`/`to_scored_docs` into a
`lucene-search/src/knn_query.rs` as `search_knn_{float,byte}_vector_query`
returning `Vec<ScoreDoc>`, reduce `vectors.rs` to handle validation and
argument decoding, and move the differential fixture with it. Stated in
`vectors.rs`'s own module doc so the next reader does not have to rediscover
it.

### 23. [CORRECTNESS] A caller error surfaced as `Decode`

A query vector of the wrong length reached `values.scorer(...)`, whose
`QueryDimensionMismatch` this module mapped to `FfiStatus::Decode` — which
tells a JNI caller "the index is corrupt", plausibly failing a shard for a bad
request. Java raises `IllegalArgumentException` ("vector query dimension: X
differs from field dimension: Y").

Resolution: **fixed.** Checked in `knn_search` before the scorer is built,
returning `InvalidArgument` with Java's own message shape, for both entry
points. The test was renamed to say which status it now proves.

### 24. [CORRECTNESS] A test asserted half of what its name claimed

`an_unknown_field_and_a_non_vector_field_are_both_invalid_arguments` only
exercised the unknown-name path; the "declared in `.fnm` but absent from
`.vemf`" branch — a real shape, since a `.fnm` lists every field, vector or
not — was never reached. The fixture segment happens to contain nothing but
vector fields.

Resolution: **fixed.** Split into two tests, the second fabricating the state
on the handle (the same test-only technique `query.rs`'s `corrupt_doc_bytes`
uses for a branch the public API cannot produce), and asserting the specific
message.

### Advisories, also addressed

- **`ffi_writer_update_binary_doc_value` had no success path.** Its only
  exercise was an error case, so a transposed `dv_field_name`/`value_ptr`
  pair in a ten-parameter C signature would have gone unnoticed. Added
  `a_binary_doc_values_update_against_a_binary_field_succeeds` over a
  BINARY-declared field, plus the mirror-image numeric-against-binary type
  error.
- **A tautological assertion.** `assert_eq!(b_seq, 0, "a refused update must
  not consume a seqNo slot")` was true by construction (`b_seq` starts at 0
  and the failing path never writes it). Reworded to what it can prove, and
  the claim it was *trying* to make is now asserted properly: after two
  refused updates, the next accepted one's seqNo is exactly `previous + 1`.
- **An overflow check that read as dead code.** `decode_document_block`
  computed a `checked_add` total and discarded it (`let _ = total;`); the
  check was load-bearing but nothing expressed the link to the unchecked
  `offset += *count` it protected. Rewritten so the running offset is itself
  the `checked_add`, and the separate loop deleted.
- **A half-specified HNSW graph pair was silently ignored.** `.vem` without
  `.vex` (or vice versa) degraded every later search to an exhaustive scan
  with no diagnostic. Now `InvalidArgument` — deliberately stricter than the
  `.nvm`/`.nvd` and `.dvm`/`.dvd` precedent, since this entry point is new and
  has no caller to keep compatible with the weaker contract.
- **A test leaked its handles.** The filter-scoring test never closed its
  segment/directory; these registries are process-wide and the same file's
  other tests assert on handle-reuse behaviour.
- **The bench's per-call denominator.** `per_thread = CONCURRENT_ITERS /
  threads` truncates, but the printed figure divided by `CONCURRENT_ITERS`.
  Harmless at 1/4/16/32 (all divide evenly, so the table above is unaffected),
  but an odd `FFI_OVERHEAD_THREADS` would have overstated the speedup. Now
  divides by `per_thread * threads`.
- **The allocation-hazard scanner's substring list.** It named
  `Vec::with_capacity`/`String::with_capacity` literally, silently exempting
  the turbofished `Vec::<T>::with_capacity` form and every other
  capacity-preallocating constructor. Widened to `::with_capacity(`, with a
  new self-test pinning the spellings it must catch and the fallible
  replacements it must not.
- **`ffi_get_last_error_message` is the one exported symbol outside `guard`.**
  Considered and **declined**, with b15's reasoning still standing and one
  addition: `guard` itself touches two more thread-locals (`ERROR_RECORDED`,
  `LAST_ERROR`), so wrapping cannot make a TLS-teardown call safer than the
  body already is — and `guard`'s backfill would destroy the very message the
  caller is retrying to read. Documented in `error.rs`, unchanged.

---

## Panic safety and `unsafe` scope, re-verified

- **98 exported `extern "C" fn`s**, every one wrapping its body in `guard`
  except `ffi_get_last_error_message` (the documented exception above).
  Verified by script over the whole crate, not by eye.
- **Every `unsafe` block in production code carries a SAFETY comment**, and
  each one was re-read against the code it now guards. Verified by script.
- **No caller-supplied length reaches a non-fallible allocation.** The
  source-scanning invariant test still passes and is now stricter (advisory
  above). The new sites go through `try_with_capacity`, and the two new
  numeric hazards this batch could have introduced are both closed: `k`
  (finding 12) and the document-block field-count total (advisory above).
- **Every new numeric crossing the boundary is validated the way b15
  validated the rest**, returning a typed `FfiStatus` with a retrievable
  message: `minimum_should_match >= 0`, `clause_occurs <= 3`,
  `clause_kinds <= 1`, `clause_parents` in `-1..i`, `clause_params == 0` for a
  TERM clause and `>= 0` for a BOOLEAN one, nesting depth, clause count,
  `k >= 1`, `similarity` in `-1..=3`, the query vector's dimension, the
  delete-query node kind/parent/depth/count, and `custom_freq >= 1`.
- **`c_char` signedness**: no `as`-cast of a `*const c_char` anywhere in the
  new code; the one byte-pointer conversion uses `.cast::<c_char>()`.
  Confirmed by a clean `cargo clippy --target aarch64-unknown-linux-gnu
  --all-targets -- -D warnings` as well as the native one.

## The ABI break, stated plainly

This batch changes the C ABI of **eleven** exported functions: the eight
boolean entry points (new clause-array parameter block), `ffi_open_segment`
(`pay_name`/`pay_name_len`), and `ffi_writer_{add_document, update_document,
delete_documents}` (`out_seq_no`). It is deliberate, it is taken in one batch
rather than spread over several, and it is taken now because the JNI wrapper
class that would have to follow it does not exist in this repo. The
boolean-query half is specifically designed so the *next* such change is not
needed.

The proof that it is behaviour-preserving for everything the old format could
express is `legacy_boolean_abi.rs`: a `#[cfg(test)]`-only bridge that
re-expresses the pre-c13 three-bucket call on the new format and forwards to
the **real exported symbol**, so the crate's whole pre-existing boolean-query
suite — matched doc sets, scores, MAXSCORE pruning, explanations, error codes,
the clause cap, the null-pointer checks — runs unchanged against the new
boundary. New capabilities are tested directly against the new signature, not
through the bridge.

## Verdict

| File | Verdict |
|---|---|
| `handle.rs` | swept-clean (finding 19: shard field, narrowed generation) |
| `registry.rs` | swept-clean (finding 19 fixed + measured) |
| `raw.rs` | swept-clean (scanner widened) |
| `segment.rs` | swept-clean (finding 13 fixed; `decode_live_docs` extracted) |
| `query.rs` | swept-clean (findings 1, 2, 3, 4, 6, 7 fixed); finding 5 open |
| `explain.rs` | swept-clean (findings 1-4, 7 fixed) |
| `directory_reader.rs` | swept-clean (findings 1-4, 7 fixed) |
| `vectors.rs` (new) | swept-clean (findings 8-12, 23, 24 resolved); finding 22 open |
| `writer.rs` | swept-clean (findings 14-17, 21 fixed); finding 18 open |
| `legacy_boolean_abi.rs` (new, test-only) | swept-clean |
| `lib.rs` | swept-clean (re-exports + deferral list corrected) |

## Open items

- **KNN query policy is at the wrong altitude** (finding 22) — move it into
  `lucene-search/src/knn_query.rs`. Blocker: `crates/lucene-search` belonged
  to a concurrent batch. Until then, no non-FFI consumer (including the
  multi-segment fan-out) can run a vector query.
- **Non-`(field, term)` clause kinds** (finding 5) — `Prefix`/`Wildcard`/
  `Regexp`/`Fuzzy`/`TermInSet` are one tag value away and need no ABI change;
  `Phrase`/`MultiPhrase`/`Boost`/`DisjunctionMax`/`ConstantScore`/`Span`/
  `PointsRange` need one more parallel array each.
- **Multiple sort fields** (b15's deferral, re-assessed) — still blocked, and
  *not* at this boundary: `lucene-search` has no `Sort`/`SortField`
  composition at all (`doc_value_query.rs` exposes single-key ascending
  sorts plus the range-then-sort path). Nothing for an FFI wrapper to wrap.
- **`apply_merge` / manual merge execution** (finding 18) — needs an FFI way
  to *run* a merge first.
- **`term_vectors_query::matched_term_offsets` has no wrapper**, so a JNI-only
  caller cannot compute the spans `ffi_assemble_fragments` consumes.
  Unchanged from b15.
- **Batch-name collision**: `LEDGER.md`'s open-work list already contains an
  unstarted item labelled "c13 — c1's caller migration". This batch was
  assigned the name `c13-ffi-surface` by the coordinator; the two are
  unrelated and one of them wants renaming.

## Gate

`cargo fmt --all --check`, `cargo clippy -p lucene-ffi --all-targets -- -D
warnings` (and the same for `--target aarch64-unknown-linux-gnu`), and
`cargo test -p lucene-ffi` all pass: **507 tests, 0 failures**.
`cargo llvm-cov -p lucene-ffi --summary-only`: every file at or above the 95%
line bar (lowest `explain.rs` 96.26%, then `directory_reader.rs` 96.58%;
`vectors.rs` 97.89%); crate total **98.19%**, up from b15's 97.97%.
