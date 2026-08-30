# c23 — `IndexWriter` indexes positions, offsets and payloads, and real Lucene reads them back

Follow-up batch. It closes the two carry-over items c20 left, which together
meant **this port had never had real Lucene read back positions it wrote**:

> No cross-engine verifier for the postings write path — this batch adds a
> whole new wire region real Lucene has never read back; the evidence is two of
> this port's readers agreeing with each other and with Java's reader on Java's
> bytes.

> `IndexWriter` still cannot index positions, so no `CheckIndex` run has seen a
> `.pos`/`.pay` this port wrote.

Java read from **`/home/tuong/work/lucene-10.5.0`**, the pinned tag, not
`/home/tuong/work/lucene` (`main`, 4 574 commits ahead — `PROTOCOL.md`, c18).
`Lucene104PostingsWriter`/`Reader` and `IndexWriter` are all on c18's
differs-between-tag-and-`main` list; nothing this batch reads from them differs
between the two, and the one line c18 reverted (`flushDocBlock`'s dense-block
rule) is untouched here.

Findings: **3 CORRECTNESS** (all fixed), **6 MISSING** (all fixed), **2 PERF**
(both measured, one fixed-by-design and one recorded with its cause and its
fix), **3 INTENTIONAL**.

`scripts/verify-write-path.sh` **21/21 → 22/22** (21/21 confirmed by running
it, not assumed).

---

## Files swept

| Rust file | Java counterpart (10.5.0) |
|---|---|
| `crates/lucene-index/src/indexing_chain.rs` | `lucene/core/src/java/org/apache/lucene/index/IndexingChain.java` (`PerField.invert`), `TermsHashPerField.java` (`add`/`writeProx`), `FreqProxTermsWriterPerField.java` |
| `crates/lucene-index/src/index_writer.rs` | `index/IndexWriter.java`, `index/IndexingChain.java` (`processField`, `startStoredFields`, the per-format consumers), `index/DocumentsWriterPerThread.java` (`flush`), `index/TermVectorsConsumerPerField.java`, and the caller side of `codecs/lucene104/Lucene104PostingsWriter.java` |
| `crates/lucene-index/src/segment_writer.rs` | `index/DocumentsWriterPerThread.flush`'s stored-fields slice — **swept, unchanged**, see below |
| `crates/lucene-codecs/src/field_infos.rs` (`write` only) | `codecs/lucene94/Lucene94FieldInfosFormat.write` + `index/FieldInfo`'s constructor |
| `crates/lucene-ffi/src/writer.rs` (`map_writer_error` only) | *(no Java counterpart: the C-ABI status mapping)* |

New files: `crates/lucene-index/examples/write_positions_segment_fixture.rs`,
`fixtures/src/VerifyPositionsSegment.java`,
`crates/lucene-index/tests/positions_write_path.rs`. Also touched:
`scripts/verify-write-path.sh` (one case), `benchmarks/rust-runner/src/index_bench.rs`
(one A/B arm), `docs/parity.md` (two rows).

---

## `crates/lucene-index/src/indexing_chain.rs`

Java: `IndexingChain.PerField.invert` — the loop that pulls
`CharTermAttribute`/`OffsetAttribute`/`PositionIncrementAttribute`/
**`PayloadAttribute`** off the token stream and hands them to
`TermsHashPerField.add`, which writes the prox stream.

### Method correspondence

| Rust | Java | Verdict |
|---|---|---|
| `invert_documents` | `PerField.invert` (no-payload path) | identical in shape; now a shorthand for the call below |
| `invert_documents_with_payloads` | `PerField.invert` + `TermsHashPerField.writeProx`'s `payload` argument | **added** (F2) |
| `PayloadContext` / `PayloadSource` | `PayloadAttribute` | **added** (F2), not-in-Java as a *type* — Java reads an attribute off the token stream, see F11 |
| `PostingEntry::payloads` | `FreqProxPostingsArray`'s prox stream payload bytes | **added** (F2) |
| `PostingEntry::has_payloads` | `FieldInfo.hasPayloads()` at the entry level | **added** (F2) |
| `InMemoryInvertedIndex::ram_bytes_used` | `TermsHash.bytesUsed` | now counts payload slots and their bytes (F9) |

Java methods with no Rust counterpart, unchanged from b9/c17: the
`TokenStream` lifecycle (`reset`/`incrementToken`/`end`/`close`), so `end()`'s
trailing position increment is dropped; `IndexableFieldType`, so there is no
`storeTermVectorPositions`-style per-field switch independent of
`IndexOptions` (F12); `invertState.maxTermFrequency`/`uniqueTermCount`, which
this port recomputes at flush instead of accumulating.

### F2 `[MISSING → fixed]` there was no payload axis at all

**Java**: any `TokenFilter` may call `PayloadAttribute.setPayload`;
`PerField.invert` reads whatever the attribute holds for the current token and
`TermsHashPerField.writeProx` writes it into the prox stream.
`FreqProxTermsWriter.flush` then hands it to
`Lucene104PostingsWriter.addPosition`, which writes the length into `.pay`'s
payload-length block and the bytes into `.pay`'s byte run.

**We did**: nothing. `Occurrence` carried position and offsets only, this
module's own doc comment said so ("payload bytes … `IndexWriter::build_postings_output`
never sets `has_payloads` or populates `payloads`"), and
`postings_writer`'s payload support — which has existed since b5 — had no
producer.

**Fixed**: `invert_documents_with_payloads(docs, analyzer, payload_fields,
source)`.

- `PayloadSource` is `&dyn Fn(&PayloadContext<'_>) -> Option<Vec<u8>>`, this
  port's stand-in for `PayloadAttribute`. `lucene_analysis::Token` has no
  payload attribute and `lucene-analysis` is not this batch's file, so the
  supplier is passed in rather than read off the token — same layering (the
  analysis side decides the bytes, the indexing chain only records them), one
  indirection instead of an attribute lookup.
- `payload_fields` is the per-field gate, because payload presence is a
  **field** property in Lucene (`FieldInfo.hasPayloads()`, one `.fnm` bit) and
  never a per-token one. "No payload on this token" and "this field has no
  payloads" are different states on the wire — only the second omits `.pay`'s
  length stream entirely — so the gate cannot be something `source` signals by
  returning `None`.
- `PostingEntry::payloads` is **either empty or exactly parallel to
  `occurrences`**, with a zero-length entry where the source returned `None`
  (Java treats a `null` payload and a zero-length one identically,
  `Lucene104PostingsWriter.addPosition:316-319`). Empty is the state for a
  field without payloads, so a non-payload field pays nothing — a `Vec<Vec<u8>>`
  per posting entry rather than per occurrence was the deliberate choice, see
  F9.

Tests: `payload_slots_are_filled_only_for_the_fields_that_declare_payloads`
(with the non-payload field in the *same batch* as the control),
`the_payload_source_sees_the_field_document_position_and_offsets`,
`invert_documents_records_no_payload_slots_at_all` (whose control is a source
that panics if it is ever reached), and
`ram_bytes_used_counts_payload_slots_and_their_bytes`.

### Verdict

Swept clean for this batch's scope. b8's structural `lucene-analysis` items are
unchanged and one of them is now **shipped rather than latent** — see F13.

---

## `crates/lucene-index/src/index_writer.rs`

Java: `IndexingChain.processField`'s fan-out into the per-format consumers,
`FreqProxTermsWriter.flush` → `Lucene104PostingsWriter`,
`TermVectorsConsumerPerField`, and `FieldInfo`'s own invariants.

### Method correspondence (what this batch changed or added)

| Rust | Java | Verdict |
|---|---|---|
| `PostingsFieldConfig::store_payloads` | `FieldInfo.hasPayloads()` as `PushPostingsWriterBase.setField` reads it | **added** (F1) |
| `resolve_postings_field`'s payload guard | `FieldInfo.checkConsistency`'s "indexed field cannot have payloads without positions" | **added** (F5) |
| `IndexWriter::set_payload_source` | *(no Java equivalent: Java installs a `TokenFilter`)* | **added** (F2/F11) |
| `IndexWriter::payload_field_names` | `FieldInfos.hasPayloads()`, per field | **added** (F2) |
| `invert_pending_fields`'s two new arguments | `PerField.invert`'s `PayloadAttribute` | **added** (F2) |
| `build_postings_output`'s `has_payloads` | `Lucene104PostingsWriter`'s `writePayloads` | **fixed** (F1) — was the constant `false` |
| `build_postings_output`'s `payloads` | `addPosition`'s `payload` argument | **added** (F2) |
| `build_term_vectors_output`'s three axis flags | `TermVectorsConsumerPerField.start`'s `doVectorPositions`/`doVectorOffsets`/`doVectorPayloads` | **fixed** (F3) — were `true`/`false`/`false` regardless of the field |
| `TermVectorFieldConfig::{index_options, store_payloads}` | `FieldType.storeTermVector*` | **added** (F3/F12) |
| `Error::{PayloadsWithoutPositions, NoPayloadFields}` | `IllegalArgumentException` from `checkConsistency` / *(no Java equivalent)* | **added** (F5) |
| `BoxedPayloadSource` / `PayloadSourceRef` | *(none)* | **added**; `Send + Sync` is load-bearing, see F10 |

Java behaviours still with no Rust counterpart here, unchanged: per-field
analyzers (`Analyzer.getWrappedAnalyzer`), `FieldInfo.setStorePayloads`'s
content-driven promotion (F11), `IndexableFieldType`'s vector axes independent
of `IndexOptions` (F12), and `invertState`'s incremental statistics.

### F1 `[CORRECTNESS → fixed]` `has_payloads` was the constant `false`, while the `.fnm` said otherwise

**Java**: `Lucene104PostingsWriter`'s constructor opens `.pay` when
`state.fieldInfos.hasPayloads() || hasOffsets()`, allocates
`payloadLengthBuffer` when `hasPayloads()`, and `PushPostingsWriterBase.setField`
takes `writePayloads` from `fieldInfo.hasPayloads()`. The **reader** mirrors it
exactly: `Lucene104PostingsReader`'s constructor opens `.pay` on the same
condition (`:151-160`) and `BlockPostingsEnum` frames every block's
payload-length run off `fieldInfo.hasPayloads()` (`:417`).

**We did**: wrote `has_payloads: false` into every `FieldPostingsInput`, with a
`// ... never sets has_payloads` note in `indexing_chain`'s module doc. But the
`.fnm` is written straight from the caller's `FieldInfo`, `STORE_PAYLOADS` bit
included. So a field declared `store_payloads: true` produced a segment whose
`.fnm` promises payloads and whose postings carry none:

- **positions + payloads, no offsets** → no `.pay` is written at all, and real
  Lucene's reader opens `.pay` unconditionally for that field. The segment
  cannot be opened.
- **positions + offsets + payloads** → `.pay` exists, but framed for offsets
  alone. Lucene reads a payload-length block where the offset start-deltas are.
  **Measured**, by reverting this one line and re-running the new verifier:

  ```
  MISMATCH body:dense doc 0 occurrence 0 offsets=23..1524 expected 0..5
  MISMATCH body:alpha doc 0 occurrence 0 payload=000000…00 expected 04050607
  MISMATCH body:sparse doc 0 occurrence 0 position=1 expected 3
  ```

  Every offset, every payload and some positions decode to garbage, on a
  segment that opens cleanly.

**Fixed**: `PostingsFieldConfig` carries `store_payloads` from the `FieldInfo`,
`build_postings_output` sets `has_payloads` from it (and `subsumes_positions()`,
belt-and-braces against F5's guard ever being relaxed), and fills
`TermPostings::payloads` by moving — not cloning — each entry's slots out of
the inverted index, which this pass already consumes.

Tests: `a_payloads_field_without_offsets_writes_a_pay_file_and_round_trips_its_bytes`
(the shape that could not be opened), with
`a_positions_field_without_payloads_writes_no_pay_file` as its negative control
so "a `.pay` exists" is evidence of something;
`the_payload_source_decides_the_bytes_that_land_on_disk`, which indexes the same
corpus twice with two different sources and requires the read-back bytes to
differ — a writer that wrote a constant passes every other assertion here and
fails that one; and
`a_payloads_field_with_no_source_still_writes_the_payload_length_stream`.
Cross-engine: `VerifyPositionsSegment` reads every sampled occurrence's payload
back through real Lucene (F7).

### F3 `[MISSING → fixed]` term vectors recorded positions only, whatever the field indexed

**Java**: `TermVectorsConsumerPerField.start` sets `doVectorPositions`,
`doVectorOffsets` and `doVectorPayloads` from the field's `FieldType`, and
`writeProx` records each occurrence's position, offset span and payload
accordingly.

**We did**: `build_term_vectors_output` hardcoded `has_positions: true,
has_offsets: false, has_payloads: false` and `start_offsets: None, end_offsets:
None, payloads: None` on every `TermVectorTerm`, for every field.
`term_vectors::write_best_speed` has supported all three axes since c8, so this
was a missing caller, not a missing format.

**Consequence, and why it was invisible**: `CheckIndex.testTermVectors` at
`MIN_LEVEL_FOR_SLOW_CHECKS` compares a field's vector against its postings
occurrence by occurrence — but only for the axes the **vector** declares
(this port's port of it is `check_index.rs`'s `term_vectors.match_postings`,
which guards on `t.start_offsets.is_some()` / `fi.store_payloads` +
`t.payloads.is_some()`). A vector that declares neither makes the offset and
payload halves of that cross-check skip silently. c9 ported those comparisons
against Java-written fixtures; on a segment this port wrote they could not fire.

**Fixed**: `TermVectorFieldConfig` carries `index_options` and
`store_payloads`; the vector records offsets when the field indexes offsets and
payloads when it stores them. Positions stay unconditional — see F12.

Tests: `term_vectors_carry_the_offsets_and_payloads_the_field_declares`, whose
control is a second term-vector field **in the same segment** declaring neither
axis, so the fix cannot be "always record them". Cross-engine:
`VerifyPositionsSegment` asserts `Terms.hasPositions()/hasOffsets()/hasPayloads()`
on the stored vector and re-checks one sampled document's whole vector against
the manifest, and real `CheckIndex` then runs its own cross-check over all
20 000. **Measured negative control**: reverting the two axis flags gives

```
MISMATCH document 0 body term vector has positions=true offsets=false payloads=false,
         expected all true (the field indexes all three)
```

and real `CheckIndex` stays **clean**, which is the point of the finding.

### F4 `[CORRECTNESS → fixed]` the writer emitted a `.fnm` it could not itself re-open, and that silently disabled every postings check

Found by doing what the brief asked — running this port's `check_index` over a
writer-produced positions segment — and it had nothing to do with positions.

**Java**: `FieldInfo`'s constructor coerces `storeTermVector`, `storePayloads`
and `omitNorms` to `false` for a field whose `indexOptions` is `NONE`
(`FieldInfo.java:110-114`) *before* `checkConsistency` looks at them. So the
three "non-indexed field cannot …" errors in `checkConsistency` are
unreachable, and a Java-written `.fnm` never has those bits set for a
non-indexed field.

**We did**: a Rust `FieldInfo` is a plain struct with no constructor, so a
caller can hand `IndexWriter::open` a stored-only field with `omit_norms: true`
— which is what a caller who writes one `field()` helper for both indexed and
stored-only fields naturally does. `field_infos::write` then put the bit
straight on the wire. Real Lucene *reads* it back fine, because the same
constructor coercion runs on the read side — which is exactly why every
cross-engine verifier stayed green, including the new one in this batch, whose
first revision wrote such a `.fnm` and passed `DirectoryReader` and `CheckIndex`
without complaint.

**This port's own reader does not coerce**: `field_infos::parse` calls
`check_consistency`, which rejects it. So `check_index`'s `fnm.open` failed, and
because `check_postings` is gated on `field_infos` being `Some`, **every
postings check in that segment was skipped without a failure being recorded**
— `all_passed()` returned `true` over a segment whose term dictionary had never
been opened. That is the failure mode this whole batch exists to rule out, one
level up.

**Fixed** in `field_infos::write`, which is where every writer in the port
funnels (flush, merge, doc-values-update generation): the three bits are
cleared for a non-indexed field, exactly as Java's constructor does. Coercing
rather than erroring is what Java does, and it is what keeps `write` → `parse`
total.

Test: `a_non_indexed_field_with_indexed_only_flags_still_writes_a_reopenable_fnm`,
whose negative control is the second half — the same three bits on an *indexed*
field must survive, so the fix cannot be "always clear them" — plus a
`check_directory` run asserting the segment is clean. And
`the_positional_offset_and_vector_checks_all_fire_on_it` (F8) exists precisely
so a future recurrence shows up as "the checks did not run" rather than as
silence.

### F5 `[MISSING → fixed]` `store_payloads` without positions was accepted until the `.fnm` was already being written

**Java**: `FieldInfo.checkConsistency` — "indexed field 'x' cannot have
payloads without positions" — raised from the constructor, i.e. before any
document is indexed.

**We did**: `field_infos::write`/`parse` catch it, but only once a commit is
half-built, and the error names the codec rather than the caller's mistake.

**Fixed**: `resolve_postings_field` rejects it with
`Error::PayloadsWithoutPositions`, so both `set_postings_field` and
`add_postings_field` (which share the resolver) fail where the field is opted
in. `set_payload_source` additionally rejects a writer where **no** declared
field stores payloads (`Error::NoPayloadFields`), because that configuration
discards every payload the source produces, silently. It is checked against the
declared field list rather than the current opt-ins so the call is
order-independent.

Tests: `set_postings_field_rejects_payloads_on_a_field_without_positions`
(both entry points) and `set_payload_source_rejects_a_writer_with_no_payload_field`
(including that clearing with `None` is always allowed). Both new variants are
enumerated in `lucene-ffi`'s `map_writer_error` as `InvalidArgument`, which c13
deliberately made a compile error to forget.

### F6 `[MISSING → fixed]` a test in this file had never run

`#[test]` appeared twice on `a_doc_values_update_still_works_against_a_merged_segment`
(once above its doc comment, once below) and not at all on the function after
it, so `update_numeric_doc_value_writes_a_generation_the_reader_can_replay` was
dead code. `rustc` says so — `duplicate_macro_attributes` plus a
`function is never used` — and both were being carried as accepted warnings.

**Fixed**: attribute moved. The test passes; the value here is that it is now
running, and that the file is warning-free so the next one is visible.

### F10 `[CORRECTNESS-adjacent → fixed]` the boxed payload source has to be `Send + Sync`

`lucene-ffi`'s `WriterHandle` carries an `unsafe impl Send`/`Sync` whose
justification is, verbatim, that `IndexWriter` "is a plain aggregate of `Vec`s,
`String`s, `SegmentInfos` and `Option`s … no interior mutability at all", and
that `read_recovering(writers())` therefore hands `&WriterHandle` to
arbitrarily many threads safely. A `Box<dyn Fn>` field falsifies that: a caller
in any crate could capture an `Rc` or a `Cell`, and `lucene-index`'s
`forbid(unsafe_code)` would not stop them.

**Fixed** by bounding `BoxedPayloadSource` with `Send + Sync`, which keeps the
existing safety argument true by construction and costs nothing — every source
in the tree captures `Copy` data or nothing. Recorded here rather than in
`registry.rs` because the fix is on this side of the boundary.

### Verdict

Swept clean for this batch's scope. `IndexWriter` now indexes all four
`IndexOptions` rungs plus offsets and payloads, and the result is read back by
real Lucene occurrence by occurrence.

---

## `crates/lucene-index/src/segment_writer.rs`

Java: the stored-fields slice of `DocumentsWriterPerThread.flush`.

**Swept, unchanged, and that is the finding.** The brief named this file as the
place to thread positions through, on c20's reading that the blocker was "in
`indexing_chain`/`segment_writer`". It is not: `flush_stored_only_segment` and
its sorted/blocks siblings write `.fdt`/`.fdx`/`.fdm` + `.fnm` + `.si` and
nothing else, by design — the module's own doc comment says so at length — and
`IndexWriter::build_and_write_segment` calls it for the stored half and then
writes postings, norms, term vectors, doc values and vectors itself, each
patching `.si` afterwards. Every consumer of the invert pass, positions
included, already lives in `index_writer.rs`. Adding a positional path here
would have duplicated `write_postings_files`, not enabled anything.

The one thing that *is* worth recording: `flush_stored_only_segment`'s name is
now the only accurate part of its contract, since c17 the sorted variant, c7
the blocks variant. No change made.

---

## The cross-engine verifier (F7)

### F7 `[MISSING → fixed]` nothing outside this port had ever read a `.pos`/`.pay` it wrote

`scripts/verify-write-path.sh`'s only whole-index cases before this batch —
`write_full_segment_fixture`, `write_merged_segment_fixture`,
`write_sorted_segment_fixture`, `write_vector_segment_fixture`,
`write_block_segment_fixture`, `write_doc_values_updates_fixture` — all index
`DocsAndFreqs`, so none of them contains a `.pos` file at all. c20 had just
added the level-0/level-1 `.pos`/`.pay` skip records to what this port writes,
with two of its own readers as the only evidence.

**Fixed**: `write_positions_segment_fixture` + `VerifyPositionsSegment`,
`verify-write-path.sh` **21/21 → 22/22**.

**The fixture is sized to the format, not to convenience.** `BLOCK_SIZE` is
256 and `LEVEL1_NUM_DOCS` is 8 192, and which path a term takes is decided
entirely by where its postings sit relative to them:

| property | value | what it reaches |
|---|---|---|
| documents | 20 000 | — |
| `dense` `docFreq` | 20 000 | **two complete level-1 spans** (8 191, 16 383), 78 level-0 blocks, a group-varint tail |
| per-document frequency | `1 + d % 5` | period **coprime with 256**, so `.pos` block boundaries drift against `.doc` ones and the level-1 `posBufferUpto` is non-zero — c20's Tier-2 review found a period-4 cycle making that byte indistinguishable from a hardcoded zero |
| `dense` `totalTermFreq` | 60 000 | 234 full `.pos` `PForUtil` blocks + a vint tail (60 000 is not a multiple of 256, asserted) |
| `dense_block_crossing_doc` | 598 | a document whose occurrences **straddle** a `.pos` block boundary; the verifier requires it to be sampled |
| payload length | `(7d + p) % 5`, 0..=4 | a non-uniform payload byte run per block, with Lucene's `null`-payload equivalent (length 0) frequent |
| `blk256` / `blk257` | `ttf` 256 / 257 | `lastPosBlockOffset`'s **third** branch (Java's `-1` sentinel: no vint tail at all) and its neighbour one occurrence over |
| `solo` / `duo` | `docFreq` 1 / 2 | the singleton the term dictionary pulses inline (no `.doc` skip data) and the smallest non-singleton |
| `gap` / `sparse` | `docFreq` 6 667 / 207 | a term below `LEVEL1_NUM_DOCS` (level-0 only) and one below `BLOCK_SIZE` (all-vint) |
| fields | 6 | `tag` (`DOCS`), `count` (`DOCS_AND_FREQS`), `title` (positions), `notes` (positions **+ payloads, no offsets**), `head` (positions **+ offsets, no payloads**), `body` (all three, plus norms and term vectors) |

The six fields share one `.doc`/`.pos`/`.pay` file set, which is what makes the
per-field framing real rather than nominal: `title`'s blocks must not read
`.pay` even though `body`'s do, `tag`'s must not read `.pos`, and `notes` and
`head` are the two ways `.pay` exists for only one of its two reasons.

**The manifest is not derived from the structure under test.**
`positions-manifest.properties` is written from an *independent* whitespace
re-scan of the very text handed to `add_document`, not from
`invert_documents`' output, and the payload rule is *recomputed in Java* as
well as compared to the manifest. Three separately-built things have to agree:
the intended token layout, this port's invert-and-encode path, and Lucene's
decoder. A manifest derived from the inverted index would agree with it however
wrong both were — which is the b4/b11 failure shape.

**What Java does with it**: opens the directory with `DirectoryReader`; checks
all six fields' `IndexOptions` and `hasPayloads()` against what the fixture
declared; checks nine terms' `docFreq`/`totalTermFreq`; asserts the fixture is
**non-degenerate** (`dense` spans two level-1 spans, fills more than two `.pos`
blocks, and does not end on a block boundary); walks 51 sampled documents — 
either side of each block and level-1 boundary, the first document of the tail,
first, last, the block-crossing document, and an irregular stride of 719 — with
a **fresh `PostingsEnum` per sample**, so each is reached through `advance` and
the skip records rather than by sequential iteration; compares every
occurrence's position, `startOffset`, `endOffset` and `getPayload()`; requires
at least three distinct payload lengths to have been observed; runs the same
comparison against the stored term vector; runs
`PhraseQuery("alpha","beta")` and its **reverse** (which must match nothing —
positions that decode individually but sit at wrong absolute values still match
every term query and no phrase); checks the `DOCS`-only field reports
`hasFreqs()/hasPositions()/hasOffsets()/hasPayloads()` all false and `freq() ==
1` for all 20 000 documents; and finally runs `CheckIndex` at
`MIN_LEVEL_FOR_SLOW_CHECKS`.

### Negative controls

Every one was run by mutating the writer, regenerating the fixture and
re-running the verifier.

| mutation | caught | first failure |
|---|---|---|
| `has_payloads: false` (F1's pre-batch state) | **yes** | `body:alpha doc 0 … offsets=5..5652 expected 22..27`, and every payload |
| term-vector `has_offsets`/`has_payloads` back to `false` (F3's pre-batch state) | **yes** (`CheckIndex` alone: **no**) | `document 0 body term vector has positions=true offsets=false payloads=false` |
| positions written as `position + 1` | **yes** | `body:alpha doc 0 occurrence 0 position=5 expected 4` |
| **level-1** `posBufferUpto` written as a constant `0` | **yes** | `body:dense doc **8192** occurrence 0 position=2 expected 0` |
| **level-0** `posBufferUpto` written as a constant `0` | **yes** | `body:dense doc **256** occurrence 0 position=2 expected 0` |
| level-0 `payloadByteUpto` written as a constant `0` | **no** — and correctly so, see F14 | — |

The two `posBufferUpto` rows are the ones this batch exists for. They are
c20's level-0 and level-1 `.pos` skip sub-fields, and the documents they fail
at — 256 and 8 192 — are exactly the first document past the first level-0
block and the first level-1 span. Real Lucene's `BlockPostingsEnum.advance`
reads this port's skip records and lands on the right occurrence.

---

## `crates/lucene-index/tests/positions_write_path.rs` (F8)

### F8 `[MISSING → fixed]` no `CheckIndex` run had seen writer-produced positions

c9 ported `checkFields`' positional and offset ordering blocks
(`postings.positions_valid:*`, `postings.offsets_valid:*`) and
`testTermVectors`' vectors-versus-postings cross-check against **Java-written**
fixtures. Nothing this port wrote had ever reached them, because
`IndexWriter`-produced segments indexed `DocsAndFreqs`.

**Fixed**: two tests over an 8 500-document writer-produced segment (past
`LEVEL1_NUM_DOCS`, same frequency-period rule as the fixture, offsets and
payloads on, term vectors on):
`our_check_index_passes_over_a_writer_produced_positions_segment` and
`the_positional_offset_and_vector_checks_all_fire_on_it`. The second is not
decoration — it is what turns F4 from a silent pass into a failure, and it is
the reason F4 was found at all.

Deliberately **not** here: driving those checks' failure arms. That is batch
`c25-check-index-coverage`'s scope; this batch gives them their first
writer-produced *input* and leaves the corruption cases to it. `check_index.rs`
is untouched.

---

## Performance

Measured with `benchmarks/rust-runner`'s `index-bench`, 50 000 documents ×
40 tokens from a 20 000-word vocabulary, postings + norms on one field — the
same corpus and shape c3/c7/c17 measure against. A new
`LUCENE_RUST_INDEX_OPTIONS` arm raises the `body` field's rung, following the
`LUCENE_RUST_VECTOR_DIM`/`LUCENE_RUST_INDEX_SORT` pattern c10 and c17 added for
exactly this purpose.

**Read the ratios, not the absolutes.** This ran alongside several other sweep
agents; the `DocsAndFreqs` arm measures 29.3 µs/doc here against c17's 21.0 on a
quiet machine, and the spread across seven interleaved rounds is 1.5x. All arms
are the same binary, interleaved round-robin, so the comparison between them is
sound; the comparison to c17's published absolute is not.

| arm | µs/doc (min of 7) | vs `DocsAndFreqs` | writer peak RSS |
|---|---|---|---|
| `DocsAndFreqs` (the c3/c7/c17 baseline shape) | 29.33 | — | 161 MB |
| `+ positions` | 30.60 | **+1.27 (+4.3%)** | 166 MB |
| `+ offsets` | 32.73 | **+3.40 (+11.6%)** | 172 MB |
| `+ payloads` (4 bytes per token) | 55.44 | **+26.1 (+89%)** | 351 MB |
| `+ payloads, all zero-length` (control) | 59.40 † | ≈ the arm above | 311 MB |

† measured in a later, noisier round against a 43.3 µs `offsets` arm in the same
run; the point of the row is that it is **not below** the arm above it.

### F9 `[PERF → measured, recorded]` positions and offsets are cheap; payload *slots* are not, and the bytes are free

Positions cost 1.3 µs/doc and offsets another 2.1. That is the answer to the
question the brief asked, and it is a good one: the positional write path is a
~12% tax on a `DocsAndFreqs` flush.

Payloads cost **26 µs/doc and 190 MB**, and the `payloads-empty` control is
what identifies the cause: attaching *no* payload bytes at all costs the same
as attaching four bytes to every token, so the cost is the **per-occurrence
slot**, not the payload data. Concretely, for this corpus:

- ~800 000 posting-entry groups each get a `Vec<Vec<u8>>` — one allocation per
  group, capacity `freq × 24` bytes;
- `build_postings_output` then builds `TermPostings::payloads`, a
  `Vec<Vec<Vec<u8>>>` per term, whose outer vector is another 24 bytes per
  posting entry (≈48 MB across the corpus);
- and all of it is freed again per flush.

That is ~40 extra allocator round trips per document, roughly doubling the
allocation count of a structure `LEDGER.md` already records as
allocation-dominated ("8.3 MB of document text becomes 78.5 MB here, 9.4x …
real Lucene pays *zero* heap objects per occurrence").

**Recorded rather than fixed, with the fix named.** Payloads are a fresh
instance of that existing ledger item, not a new one, and the contained fix is
not available inside this batch's files: it means giving
`postings_writer::TermPostings::payloads` a flat `(bytes: Vec<u8>, lengths:
Vec<u32>)` shape instead of `Vec<Vec<Vec<u8>>>`, and having `PostingEntry`
accumulate into the same shape — two allocations per group instead of one plus
one per non-empty payload, and four bytes per occurrence instead of
twenty-four. Doing half of it (flattening only `PostingEntry`) would be
*slower*, because `build_postings_output` would have to re-materialize the
nested form. Making the change in `postings_writer` is a codec-side API change,
which is why it is handed off rather than done here.

Three things bound the exposure meanwhile: the cost is strictly per-occurrence
of a `store_payloads` field, it is zero for every field that does not declare
payloads, and payload fields are rare in practice (OpenSearch uses them for
`delimited_payload`/`term_frequency` mappings, not for `text` by default).

---

## Findings that are deliberate, or belong elsewhere

### F11 `[INTENTIONAL]` payload declaration is explicit here, content-driven in Java

Java **promotes** a field to `storePayloads` the first time the indexing chain
sees a token carrying one (`FieldInfo.setStorePayloads`), so the `.fnm` bit is
derived from document content, per segment. Here it is declared up front on the
`FieldInfo`, like every other per-field opt-in on this facade (postings, norms,
doc values, vectors, term vectors).

The wire result is identical for a field that is declared and used. The only
difference is a field declared with payloads and never given any: this port
writes an all-zero-length payload stream where Lucene would have written none.
That reads back as "no payload on any occurrence" either way, and it is
**required** for self-consistency here — the `.fnm` bit is already on disk, so
the `.pay` stream real Lucene frames from it has to exist
(`a_payloads_field_with_no_source_still_writes_the_payload_length_stream`).

Auto-promotion would mean the `.fnm` this writer emits depends on the documents
buffered when a flush happens, which is a bigger change than this batch needs
and one with its own `verifySameSchema` consequences across segments.

### F12 `[INTENTIONAL]` term-vector axes come from `IndexOptions`, and positions stay unconditional

Java's `FieldType.storeTermVectorPositions`/`Offsets`/`Payloads` are three
flags fully independent of `IndexOptions` — the `.fnm` carries only the single
`STORE_TERMVECTOR` bit, and which axes a vector holds is recorded in the `.tvd`
chunk itself. This facade has no `FieldType`, so the field's own
`IndexOptions`/`store_payloads` stand in.

That mapping is not merely the available one, it is the useful one:
`CheckIndex.testTermVectors` cross-checks a vector against its postings for
every axis both carry, so deriving the vector's axes from the postings' is what
makes that check bite (F3).

Positions stay unconditional, as they were before this batch. The analyzer
resolves a position for every token of every field regardless of
`IndexOptions`, and a term vector with positions over a `DOCS`-only field is a
legal Lucene index — `IndexingChain.verifyFieldType` forbids only vector
payloads without vector positions, and any vector axis without
`storeTermVectors`.

### F13 `[CORRECTNESS — recorded, owner `lucene-analysis`]` the offsets this port now writes are byte offsets

`lucene_analysis::Token`'s `start_offset`/`end_offset` are **UTF-8 byte**
offsets; Lucene's `OffsetAttribute` is UTF-16 code-unit offsets into the same
text. b8 raised this as one of four structural `lucene-analysis` items and
`indexing_chain::Occurrence`'s doc comment calls it "a real, currently-latent
unit caveat".

**It is no longer latent.** Those offsets now reach `.pos`/`.pay` and are read
back by real Lucene's `startOffset()`/`endOffset()`. They coincide for ASCII
and diverge for anything else, and **nothing on either side catches it**:
`CheckIndex` checks only that offsets are non-decreasing, that `endOffset >=
startOffset` and that both are non-negative — never that they index the stored
text. A highlighter slicing a Java `String` with them would cut in the wrong
place for any non-ASCII document.

The fixture is deliberately pure ASCII, which is what lets its manifest state
offsets in closed form; making it non-ASCII would not have failed anything,
which is precisely the finding. Not fixed here: the unit is decided in
`lucene-analysis`'s tokenizer, this batch's files only carry it. Recorded in
`build_postings_output`'s doc comment where the values are forwarded, and
carried below.

### F14 `[INTENTIONAL — confirmed from the other side]` level-0/level-1 `payloadByteUpto` is genuinely unobservable

c20's F9 argued from the Java source that the `payloadByteUpto` sub-field is
parsed and then always overwritten from the landing block's own decoded payload
lengths, so its stored value cannot matter. This batch tested that claim
against real Lucene rather than against a reading of Java: writing the level-0
`payloadByteUpto` as a constant `0` and re-running the verifier **passes**,
where the sibling `posBufferUpto` mutation fails at doc 256. c20's F9 is
confirmed empirically, and the field remains written correctly (it costs
nothing and the fields after it are not self-delimiting).

---

## Gates

- `cargo fmt --all` — clean.
- `cargo clippy --workspace --all-targets -- -D warnings` — **exits zero**.
  The three `clippy::type_complexity` errors the coordinator flagged as
  blocking the workspace gate were this batch's own boxed payload source, now
  factored into `BoxedPayloadSource`/`PayloadSourceRef`;
  `duplicate_macro_attributes` and a dead test (F6) were fixed on the way. The
  one remaining warning in the crate mid-batch (`unused variable: doc_order`,
  `merge.rs:3677`) was `c22-sorted-merge`'s and is gone as of the final run.
- `cargo test -p lucene-index` — **631 lib + 26 integration passed, 0 failed**,
  including this batch's 11 new ones (4 in `indexing_chain`, 5 in
  `index_writer`, 2 in `tests/positions_write_path.rs`) and F6's resurrected one.
- `python3 scripts/check-parity.py` — ok. `docs/parity.md`'s
  `IndexingChain`-inverted-index row and its `Lucene94FieldInfosFormat.write`
  row both updated.
- `python3 scripts/check-arith-allows.py` — ok. The two new test-support files
  opt out at their own file boundary with the reason c19's convention asks for;
  nothing else this batch adds does arithmetic on a file-derived value.
- `scripts/verify-write-path.sh` — **22/22**, run after every change above.
  21/21 confirmed by running it first, not assumed.
- `cargo test -p lucene-codecs -p lucene-search -p lucene-ffi` — all pass
  (1 146 in `lucene-codecs` alone). Mid-batch, `lucene-codecs` had two failures
  from another batch's in-flight arithmetic-gate audit and `lucene-search` did
  not compile at all on a `terms_dict::TermsDictEntry` field being added
  elsewhere; both settled before the final run. `field_infos`' own 37 unit
  tests and 3 fixture tests pass, which is this batch's touch there.

### A note on the shared tree

This batch ran alongside several others. `lucene-index` was transiently
un-buildable twice on a `per_source_dv!` macro mid-edit in a file this batch
also owns, `lucene-codecs` was un-lintable for most of the batch while its
arithmetic-gate burn-down was in flight, `lucene-search` did not compile for a
stretch, and `check-arith-allows.py` failed on a stale module count another
batch then fixed. Every gate above was re-run to green after those settled;
none of the breakage was caused by this batch, and none of it was fixed by it
either.

---

## Coverage

`cargo llvm-cov -p lucene-index --summary-only`, run with an **isolated
`CARGO_TARGET_DIR`** per c19's caveat that concurrent batches otherwise poison
each other's `.profraw` files.

| file | lines | note |
|---|---|---|
| `indexing_chain.rs` | **97.06%** | all 8 missed lines are `assert!` message arms in the new tests, evaluated only on failure |
| `segment_writer.rs` | **99.49%** | unchanged by this batch |
| `index_writer.rs` | **85.63%** | **below the bar, and not this batch's** — see below |
| `lucene-index` total | 92.30% | |

`index_writer.rs` was 98.10% at c17 and this snapshot has 1 319 missed lines.
Attributing every one of them to its enclosing function:

| missed lines | function | owner |
|---|---|---|
| 415 | `execute_merge` | `c22-sorted-merge`, in flight |
| 97 | `build_vectors_output` | c10 |
| 94 | `apply_packets_to_segment` | c7 |
| 76 | `resolve_delete_query` | c7 |
| 57 | `open_segment_for_deletes` | c7 |
| 50 | `build_custom_freq_postings_output` | b15 |
| ~460 | the doc-values builders, `write_*_files`, `open_dv_generation`, `collect_dense_column`, `apply_merge`, `resolve_term_span` | c6/c14/c17 |

**None is this batch's.** The lines this batch added are covered: of the
missed set, five fall in `build_postings_output` and are the closing braces of
pre-existing `if has_positions`/`if has_offsets` blocks (llvm-cov region
artifacts, not branches), and one was a genuinely dead `positions: None` arm in
`build_term_vectors_output` left by making positions unconditional — removed
rather than tested, since a coverage number cannot tell dead code from untested
code. Every other line of `set_payload_source`, `payload_field_names`,
`resolve_postings_field`'s payload guard, `invert_pending_fields`'
source plumbing, `build_postings_output`'s payload branch and
`build_term_vectors_output`'s axis derivation is exercised.

Read the file-level figure as a shared-tree snapshot, not as a verdict on this
batch: three other batches are adding to `index_writer.rs` concurrently, and a
pre-batch baseline is not obtainable from a working tree carrying all of their
uncommitted work.

---

## Carry-over items raised by this batch

- [ ] **Payload slots cost ~26 µs/doc and ~190 MB per 50 000 documents, and the
      cost is the slot, not the bytes** (F9). The fix is a flat
      `(bytes, lengths)` representation in **both**
      `postings_writer::TermPostings::payloads` and
      `indexing_chain::PostingEntry::payloads` — doing only the second is
      slower. `postings_writer` is a codec-side API, so it is a cross-file
      change this batch is not scoped to. An instance of `LEDGER.md`'s existing
      block-pool item, now with a number.
- [ ] **The offsets this port writes are UTF-8 byte offsets where Lucene's are
      UTF-16 code-unit offsets** (F13), and as of this batch they are shipped
      rather than latent. Nothing catches it: `CheckIndex` never compares an
      offset against the text it indexes. Owner is `lucene-analysis`'s
      tokenizer; b8's structural item, now with a consumer.
- [ ] **`FieldInfo` is a plain struct where Java's is a validating
      constructor** (F4). `field_infos::write` now applies the one coercion
      that was actively producing unopenable-by-us files, but the general shape
      remains: a caller can build a `FieldInfo` combination Java makes
      unrepresentable and only find out at `parse` time, or not at all. A
      constructor (or a `FieldInfo::new` that returns `Result`) would close the
      class; it touches every construction site in the workspace.
- [ ] **Payload presence is declared, not promoted** (F11). If a caller ever
      needs Java's `setStorePayloads` semantics — the `.fnm` bit derived from
      whether any token actually carried a payload — it needs the `.fnm` to be
      written after the invert pass rather than before, plus a `verifySameSchema`
      story across segments.
