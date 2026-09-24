# M3 — A Rust-written index that real Lucene can read

> **Goal:** real Java Lucene opens a full, non-toy, Rust-written index with
> `DirectoryReader.open`, passes its own `CheckIndex`, and returns hit lists
> and scores identical to this port's searcher.

| | |
|---|---|
| **Effort** | L — the largest remaining chunk of pure Lucene-format work |
| **Depends on** | [M1](m1-performance-gate.md) passing |
| **Unblocks** | [M4](m4-write-path-hardened.md) |
| **Runs in parallel with** | [M2](m2-opensearch-read-path.md) |
| **Status** | ✅ **delivered 2026-09-24** — see [Outcome](#outcome) |

---

## Why this milestone exists

This closes the largest correctness risk in the project.

`fixtures/src/` contains twelve `Verify*.java` programs — the reverse
direction of differential testing, where Rust writes bytes and real Lucene
reads them back:

`VerifyCompoundFormat`, `VerifyDocValues`, `VerifyFieldInfos`, `VerifyFst`,
`VerifyLiveDocs`, `VerifyNorms`, `VerifyPoints`, `VerifySegmentInfo`,
`VerifySegmentInfos`, `VerifySparseNumericDocValues`, `VerifyStoredFields`,
`VerifyTermVectors`.

**None of them covers `.doc`, `.tim`, `.tip` or `.tmd`.** There is no
`VerifyPostings`, no `VerifyBlockTree`, and no
`crates/lucene-codecs/examples/write_postings_fixture.rs`.

The term dictionary and postings are the format everything else hangs off.
Their write path is validated entirely by round-tripping through
`blocktree::open` and `postings::DocInput::open` — this port's *own* reader.
`docs/parity.md` is candid that this is the design:

> Does not duplicate the read side's decode logic — it only emits bytes;
> correctness is proven entirely by round-tripping through the *existing,
> unmodified* `blocktree::open`/`postings::DocInput::open`.

That argument is strong against *writer* bugs and useless against *shared
misreadings of the spec*. If the reader misunderstands a field and the writer
misunderstands it identically, every test passes and real Lucene rejects the
output. Every other major format in this port has a Java-side check that would
catch exactly that. The most important one does not.

The second half of the milestone is scope. `docs/parity.md` describes the
writer as, verbatim:

> **write path started, narrowly scoped to a single field, single `.tim`
> block, term-frequency-only postings.**

with five explicit limits: one field per call, one physical leaf block and one
trie node, `docFreq < 256` for every term, `IndexOptions::Docs`/`DocsAndFreqs`
only, and no `.pos`/`.pay` at all. The *reader* handles multi-level tries,
floor blocks, compressed blocks and full positions against real Lucene
fixtures. The writer is a proof of concept behind it.

---

## Scope

### In scope

- Java-side verification of the postings and term-dictionary write path.
- Generalising the blocktree writer to the shapes the reader already handles.
- Generalising the postings writer to full block encoding and positional data.
- An end-to-end verifier: real Lucene opens a complete Rust-written index and
  runs queries against it.
- Resolving the `.si` index-sort format divergence, one way or the other.

### Out of scope

- Concurrency, crash safety, and merge-time correctness — those are
  [M4](m4-write-path-hardened.md).
- Any OpenSearch integration.
- Vector formats. HNSW is a separate scope decision recorded in M4.
- Performance of the write path. Correctness first; M4 and M6 measure.

---

## Tasks

### T3.1 — Close the verification gap at the *current* scope, first

Do this before touching the writer. It is the cheapest possible way to find
out whether the existing bytes are right.

- Add `crates/lucene-codecs/examples/write_postings_fixture.rs`, following the
  established pattern of the ten existing `write_*_fixture` examples.
- Add `fixtures/src/VerifyPostings.java`: open the Rust-written `.doc`/`.tim`/
  `.tip`/`.tmd` with real Lucene, enumerate every term, and assert term bytes,
  `docFreq`, `totalTermFreq`, and the full doc-id/frequency postings list.
- Cover the shapes the writer already claims: mixed singleton and multi-doc
  terms, the `docFreq == 1` pulsed-into-the-term-dictionary case, and
  `IndexOptions::Docs`-only `totalTermFreq` aliasing.
- Wire it into `scripts/verify-write-path.sh` from
  [M0](m0-ci-and-green-tree.md), so it runs in CI from this point onward.

**Expect this to find bugs.** `parity.md` notes that the per-term metadata
always takes the "absolute-ish `docStartFP` delta" branch of `decodeTerm`'s
encoding and never the zigzag-singleton-delta branch — a deliberate
simplification that has never been checked against a real decoder. Finding a
problem here, at single-field/single-block scale, is enormously cheaper than
finding it after T3.2 and T3.3 have built on top of it.

### T3.2 — Generalise the blocktree writer

Target: `crates/lucene-codecs/src/blocktree.rs` and
`crates/lucene-codecs/src/postings_writer.rs`.

Remove the four structural limits, in this order:

1. **Multi-field `.tmd`.** Today `numFields = 1`. Real segments carry every
   indexed field in one term dictionary.
2. **Block splitting.** Today every term must fit one physical leaf block. Real
   `Lucene103BlockTreeTermsWriter` splits past its `minItemsInBlock` /
   `maxItemsInBlock` thresholds. This is the change that makes term dictionaries
   of realistic size possible at all.
3. **Floor sub-blocks.** `PLAN.md` already contains an audit of this
   ("Floor sub-blocks in blocktree writer: audited, no writer exists to add
   them to") — that writer now needs to exist.
4. **Multi-level `.tip` tries.** Today the index is one root
   `SIGN_NO_CHILDREN` node with `hasTerms` set and no floor data.

**The reader is the specification.** Every one of these shapes is already
decoded correctly against real Lucene fixtures —
`blocktree_multilevel_index`, `blocktree_deep_nesting_index` (4+ levels),
`blocktree_child_strategies_index`, `blocktree_compressed_index`. Write
against what the reader proves, and verify with T3.1's harness extended to
each shape as it lands.

### T3.3 — Generalise the postings writer

Target: `crates/lucene-codecs/src/postings_writer.rs`, with
`crates/lucene-codecs/src/for_util.rs` supplying the encode side.

1. **Full 128-value blocks.** Today only the group-varint tail block is ever
   written — the `flushDocBlock(true)` branch that never reaches
   `docBufferUpto == BLOCK_SIZE` — and any term with `docFreq >= 256` is
   rejected with `Error::DocFreqTooLarge`. Implement the `ForUtil`/`PForUtil`
   bit-packed path so real term distributions can be written.
2. **Skip data and impacts.** Neither is produced today. Both are required for
   the reader-side pruning that M1 measures, and impacts are what let real
   Lucene's own MAXSCORE work over these segments.
3. **`.pos` and `.pay`.** Today `IndexOptions` beyond `DocsAndFreqs` is
   rejected with `Error::UnsupportedIndexOptions`. Positions, offsets and
   payloads must all be written — the reader already handles all of them,
   including `DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS` and payloads.

The encode side of `for_util.rs` is the mirror of a decode path that is
already fixture-verified against real Lucene, which makes it the
best-constrained part of this task. Skip data and impacts are the least
constrained — verify them hardest.

When these land, `Error::DocFreqTooLarge` and `Error::UnsupportedIndexOptions`
should no longer be reachable, and the corresponding tests should be replaced
rather than deleted.

### T3.4 — `VerifyIndex.java`: the end-to-end proof

The milestone's headline artifact. Everything above is a component check; this
is the one that matters.

- A Rust example builds a complete index through `lucene_index::IndexWriter` —
  the real path through `indexing_chain.rs` and `segment_writer.rs`, not a
  hand-assembled set of files.
- ≥3 fields: an analysed text field with positions and offsets, a keyword
  field, and a numeric doc-values field.
- ≥100k documents, across multiple segments.
- At least one term with `docFreq` well above `BLOCK_SIZE` (256), so the
  bit-packed block path is genuinely exercised — a corpus of unique terms would
  silently test nothing.
- `VerifyIndex.java` then:
  1. opens it with `DirectoryReader.open`,
  2. runs real Lucene's `CheckIndex` and asserts zero errors,
  3. executes a fixed ≥50-query set spanning term, boolean, phrase and range,
  4. compares top-50 doc IDs and BM25 scores against this port's searcher.

Score comparison needs a tolerance — 1e-5 — because `PLAN.md` §3 requires
`f32` math in the same order but does not guarantee bit-identical results
across compilers. A *systematic* divergence, as opposed to last-bit noise, is a
scoring bug and must be treated as one.

### T3.5 — Resolve the `.si` index-sort divergence

`docs/parity.md` and `PLAN.md` both record this honestly:

> the `.si` index-sort byte encoding remains this port's own internal format,
> **NOT** verified byte-compatible with real Lucene's
> `Lucene99SegmentInfoFormat` (no real-Lucene-written sorted-segment `.si`
> fixture exists to derive the true `SortFieldProvider` wire format from).

This is the one place in the port where a written format is knowingly
divergent. An index-sorted segment written by this port may be unreadable by
Java Lucene, which contradicts the entire compatibility premise.

Two acceptable outcomes, and this milestone must pick one:

- **Resolve it.** Add a `Gen*.java` generator producing a real
  Lucene-written index-sorted segment, derive the true `SortFieldProvider`
  wire format from those bytes, and make the writer match. This is the
  preferred outcome and the generator is not difficult — `IndexWriterConfig
  .setIndexSort` is a one-line call.
- **Fence it off.** If the format proves impractical to match, record
  index-sorted segments as explicitly unsupported-for-interop in
  `docs/parity.md`, and make the writer refuse to emit one rather than emit a
  divergent one silently.

Leaving it as-is is not an acceptable outcome for this milestone.

---

## Acceptance criteria

- [x] `VerifyPostings.java` exists, covers the writer's full supported shape
      set, and runs in CI. *Delivered as the postings walk inside
      `VerifyIndex.java` rather than a separate class: every term of a
      120 000-document index (140 311 terms: common and rare, with and without
      positions, offsets and payloads, singletons included) is compared with
      expectations computed from the generated text. `VerifyPositionsSegment`
      already covered the occurrence-level walk through the skip data.*
- [x] Real Lucene reads a Rust-written index with **≥3 fields, ≥100k docs**, at
      least one term above `BLOCK_SIZE` (256) docs, and positions, offsets and
      payloads indexed. *120 000 documents, five segments, four fields.*
- [x] Real Lucene's own **`CheckIndex` reports zero errors** on that index
      (`MIN_LEVEL_FOR_SLOW_CHECKS`).
- [x] Across a **≥50-query set**, top-50 doc IDs match **exactly** and scores
      match within **1e-5** between Java Lucene and this port. *62 queries:
      term, conjunction, disjunction, cross-field, phrase and doc-values
      range; largest score difference 1.2e-7. The range queries are over
      doc values, because `IndexWriter` has no points write path at flush.*
- [x] The blocktree writer produces multi-field, multi-block, floor-blocked,
      multi-level output, and each shape is verified from the Java side.
      *Stronger than asked: `blocktree_writer_identity.rs` requires
      byte-identical `.tim`/`.tip`/`.tmd` against four real Lucene term
      dictionaries, one per shape.*
- [x] `Error::DocFreqTooLarge` and `Error::UnsupportedIndexOptions` are no
      longer reachable. *`DocFreqTooLarge` no longer exists.
      `UnsupportedIndexOptions` is now raised only for `IndexOptions::None`
      handed straight to the codec-level API; `IndexWriter` rejects that
      earlier with its own error, so no index write reaches it.*
- [x] The `.si` index-sort divergence is either resolved against a real Lucene
      fixture or explicitly fenced off. *Resolved in the M2 sweep (b11): the
      `SortFieldProvider` encoding is byte-verified both ways, and
      `VerifySortedSegment` runs `CheckIndex.testSort` on sorted flushes and
      merges.*
- [x] `docs/parity.md` no longer describes the postings/blocktree writer as
      narrowly scoped, or states precisely and truthfully what remains.
- [x] Per-file line coverage stays ≥95% (`AGENTS.md` invariant #8) across every
      file this milestone touches.

---

## Outcome

Most of T3.1–T3.3 and T3.5 had already landed through the August–September
sweeps (the `.psm` file, impacts, full `ForUtil` blocks, level-0/level-1 skip
data, `.pos`/`.pay` with payloads, the `.si` sort encoding) — this file still
read "not started" when the milestone was picked up. What remained, and what
this milestone delivered, per the port → benchmark → optimise workflow
(`docs/porting-workflow.md`, adopted during it):

1. **The term-dictionary writer** (`lucene-codecs/src/blocktree_writer.rs`): a
   closest-to-Java port of `Lucene103BlockTreeTermsWriter.TermsWriter` and
   `TrieBuilder`, plus `LowercaseAsciiCompression.compress` and the full
   `encodeTerm`. Byte-identical to Java on four real term dictionaries.
   Benchmarked against Lucene's own writer on identical input
   (`scripts/bench-micro.sh --bench term_dict_write`): first 0.65×/0.72×;
   restructuring the postings writer to stream one term at a time — which is
   also how Java does it — brought it to **1.92× (1M ids) and 1.30× (200k
   words)**.
2. **The end-to-end proof** (`crates/lucene-search/examples/write_verify_index.rs`
   + `fixtures/src/VerifyIndex.java`), wired into
   `scripts/verify-write-path.sh`. Its first version passed a deliberately
   broken `encodeTerm` — the corpus had no singleton terms — so it now carries
   a unique-id field and rare words, and fails on that defect.

3. **A scoring bug the proof found.** Once the corpus gained a field with
   freqs but no norms, every score on it came out 12% below Lucene's (same
   hits, same order). Java scores a norm-less document at norm 1 but against
   the collection's real `avgdl`; this port forced `avgdl` to 1 too. Fixed in
   `DirectoryReader::field_norms` (`FieldNorms::unnormed`); `parity.md`'s
   BM25 row has the detail.
4. **Review follow-ups**: two more byte-identity tests re-write real Lucene
   term dictionaries with freqs, positions, offsets, payloads and skip data
   from the term states Lucene recorded (`postings_writer`'s
   `term_dictionary_*_is_byte_identical`), closing what the all-singleton
   fixtures could not see; each was shown to fail on a seeded defect.

Whole-indexing throughput is unchanged by the streaming rewrite:
`scripts/bench-micro.sh --bench index`, A/B on one machine, 2.65× Lucene at
`0401891` (before M3) and 2.64× after (noise floor 1.14–1.40×; the 3.14× in
`docs/benchmarks/sweep-2026-09.md` was measured on a different machine).

What these checks cannot see: `maxItemsInBlock` 48→49 changes none of the
identity fixtures; VerifyIndex writes without merging (merged segments are
covered by the older `write_merged_*` cases), compares ties at rank 50 in
exact order, and has no points field.

Left for later, recorded in `docs/parity.md`: the `bitsPerValue == 0` and
dense-bitset `.doc` block encodings the real writer sometimes chooses (readers
accept the plain `ForUtil` shape this writer emits), and a points write path at
flush (range queries in the proof are over doc values).

---

## Risks and unknowns

- **T3.1 finds a foundational bug.** This is the *hoped-for* outcome, but it
  can invalidate work built on the current byte layout. It is the reason T3.1
  is sequenced first and alone.
- **Skip data and impacts are the least-constrained code in the task.** Unlike
  the bit-packing kernels, they have no fixture-verified decode mirror in this
  port to check against. They need dedicated Java-side verification, not
  round-trip testing.
- **Block-splitting thresholds change output bytes.** `minItemsInBlock` /
  `maxItemsInBlock` choices alter the physical layout without altering
  correctness. Real Lucene's reader accepts any valid split, so do not chase
  byte-identity with Java's output — chase readability, and say so in
  `parity.md`. *(Resolved better than planned: the term
  dictionary came out byte-identical to Java's once the writer was a
  closest-to-Java port. `.doc` still differs by design — no `bitsPerValue == 0`
  or dense-bitset blocks.)* This mirrors the resolution already reached for the FST builder
  in `PLAN.md` §4 risk #1.
- **Corpus realism.** A 100k-document corpus of unique terms exercises none of
  the interesting paths. Term frequencies must be Zipfian, and the acceptance
  criterion about `docFreq > 256` exists to force that.

---

## Exit artifacts

- `crates/lucene-codecs/examples/write_postings_fixture.rs`
- `fixtures/src/VerifyPostings.java`
- `fixtures/src/VerifyIndex.java` and its Rust index-building example
- A generalised `postings_writer.rs`, `blocktree.rs`, and `for_util.rs` encode
  path
- A `Gen*.java` sorted-segment generator, or an explicit unsupported record
- An updated `docs/parity.md` write-path section
