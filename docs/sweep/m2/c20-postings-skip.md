# c20 — the `.pos`/`.pay` skip pointers: `advance(doc)` stops walking the doc list

Follow-up batch. One item, carried by c15 (§F10) and recorded in `LEDGER.md`:
`.doc`'s level-0 and level-1 skip records carry the `.pos`/`.pay` file
pointers, and this port parsed them and threw them away.

Java read from **`/home/tuong/work/lucene-10.5.0`**, the pinned tag, not
`/home/tuong/work/lucene` (which is `main`, 4 574 commits ahead — see
`PROTOCOL.md` and c18). Both classes this batch touches are on c18's
differs-between-tag-and-`main` list, so both were diffed again:

| Rust file | Java counterpart (10.5.0) | tag vs `main` |
|---|---|---|
| `crates/lucene-codecs/src/postings.rs` | `lucene/core/src/java/org/apache/lucene/codecs/lucene104/Lucene104PostingsReader.java` | differs only in `checkIntegrity(MergePolicy.OneMerge)` and stale-comment fixes — nothing this batch reads. In 10.5.0 there is **one** `BlockPostingsEnum`; the `BlockDocsEnum`/`BlockImpactsEnum`/`EverythingEnum` split named in the brief was collapsed before this tag, and `skipPositions`/`posPendingCount`/`lastPosBlockFP`/`level0BlockPosUpto` all live on that one class |
| `crates/lucene-codecs/src/postings_writer.rs` | `.../Lucene104PostingsWriter.java` | the one difference is `flushDocBlock`'s dense-block encoding rule, which c18 already reverted to the tag's `numBitsNextBitsPerValue <= docRange`. This batch does not touch that line; it appends to the same `if (writeFreqs)` region a few lines above |

Findings: **3 MISSING** (all fixed), **1 CORRECTNESS** (fixed), **3 PERF**
(all fixed, two of them measured), **2 INTENTIONAL** — plus one gating and
five advisory items from the Tier-2 review, all resolved (see the review
section; the gating one is a test that could not have failed).

---

## `crates/lucene-codecs/src/postings.rs`

Java: `Lucene104PostingsReader.BlockPostingsEnum` — `reset`, `skipLevel1To`,
`doMoveToNextLevel0Block`, `readLevel0PosData`, `seekPosData`, `skipLevel0To`,
`advanceShallow`/`doAdvanceShallow`, `advance`, `skipPositions`,
`refillPositions`, `refillLastPositionBlock`, `refillOffsetsOrPayloads`,
`accumulatePendingPositions`, `accumulatePayloadAndOffsets`, `nextPosition`.

### Method correspondence (what this batch changed or added)

| Rust | Java | Verdict |
|---|---|---|
| `PosSkip` / `read_pos_skip` | `readLevel0PosData`, and the byte-identical block inside `skipLevel1To` | **added** (F1) — was inline `let _pos_end_fp_delta = …` in two places |
| `PosCursorState` + `LazyDocsCursor::{level0_pos, level1_pos, block_pos_origin}` | `level0PosEndFP`/`level0BlockPosUpto`/`level0PayEndFP` and the level-1 four, plus `skipLevel0To`'s `posFP`/`posUpto`/`payFP` locals | **added** (F1) |
| `LazyDocsCursor::skip_level1_to`'s `level0_pos = level1_pos` | `skipLevel1To`'s first four assignments | **added** (F1) |
| `LazyDocsCursor::advance_shallow`'s per-iteration snapshot | `skipLevel0To`'s loop head + its trailing `seekPosData(posFP, posUpto, payFP, payUpto)` | **added** (F1) |
| `LazyDocsCursor::position_origin` | `seekPosData`'s arguments + `accumulatePendingPositions`' `sumOverRange(freqBuffer, posDocBufferUpto, docBufferUpto)` | **added** (F1) |
| `walk_document_occurrences` | `skipPositions` + `nextPosition`/`startOffset`/`endOffset`/`getPayload` for one document | **added** (F1) |
| `read_occurrences_for_doc` | `advance(doc)` then the above — `PostingsOffsetStrategy.getOffsetsEnum`'s whole shape | **added** (F1) |
| `refill_full_position_block` | `refillPositions`' full-block branch + `refillOffsetsOrPayloads` | **extracted** (F5); was inline in `walk_wanted_occurrences` |
| `refill_last_position_block` | `refillLastPositionBlock` | **extracted** (F5); was inline, twice |
| `last_pos_block_fp` | `reset`'s three-way `lastPosBlockFP` rule (`Lucene104PostingsReader.java:526-532`) | **added** (F3) |
| `PositionBlock` | `posDeltaBuffer`/`payloadLengthBuffer`/`offsetStartDeltaBuffer`/`offsetLengthBuffer`/`payloadBytes` | **added** (F5) — Lucene's instance fields, held for the walk |
| `walk_wanted_occurrences` | *(no Java equivalent: a batch, where Java has a cursor)* | unchanged in shape; its block decode is now the shared one (F5, F8) |
| `blocktree::FieldTerms::occurrences_for_doc` | `advance(doc)` + walk | **reimplemented** on the skip path (F1) |

Java methods still with no Rust counterpart, unchanged from b5/c8/c15:
`impacts()`'s separate `needsImpacts` axis, `postings()`'s `reuse` recycling,
`intoBitSet`/`nextPostings`/`docIDRunEnd` and the rest of F10-in-b5's
bulk/lifecycle list.

### F1 `[MISSING → fixed]` `readLevel0PosData` / `seekPosData` / `skipPositions` were not ported at all

**Java**: every `.doc` level-0 block header and level-1 span entry of a
positions-indexing field carries `posEndFPDelta` (vlong) + `posBufferUpto`
(byte) and, when the field also has offsets or payloads, `payEndFPDelta`
(vlong) + `payloadByteUpto` (vint). `BlockPostingsEnum` accumulates them:
`skipLevel1To` copies the level-1 pair down to level 0 at the top of every
iteration, `skipLevel0To` snapshots level 0 *before* reading each candidate
block's header and hands the surviving snapshot to `seekPosData`, which seeks
`.pos`/`.pay` and sets `posPendingCount`. `nextPosition` then adds only the
current 256-document block's frequencies (`accumulatePendingPositions`) and
`skipPositions` steps forward. Java never reads a document it skipped.

**We did**: `read_full_block_header` and `read_level1_entry` parsed the four
fields into `let _ = …` bindings and dropped them, saying so in a comment.
Every positional accessor therefore started at the term's `pos_start_fp` and
addressed `.pos` by a running sum over the whole frequency list — which is
also why `occurrences_for_doc` had to decode the term's entire doc list before
it could locate one document. c15 measured the consequence and named it: 18.3 ms
to highlight one document of a five-million-document term, of which the
doc-list decode alone was 5.69 ms and the frequency sum most of the rest.

**Fixed**, following Java's state machine exactly:

- `PosSkip` (one record's four fields) and `read_pos_skip`, shared by the
  level-0 and level-1 readers — they read the same bytes in Java too.
- `PosCursorState` on `LazyDocsCursor`, three of them: `level0_pos`,
  `level1_pos`, and `block_pos_origin` (Java's `skipLevel0To` locals, which it
  can keep on the stack because its `nextPosition` runs inside the same object;
  here the walk is a separate call, so the snapshot has to live on the cursor).
- `position_origin()`, which is `seekPosData`'s three arguments plus the
  in-block frequency sum, i.e. everything a position walk needs to start.
- `walk_document_occurrences` — `skipPositions`' whole-block loop, then
  `nextPosition`'s per-occurrence emit — and `read_occurrences_for_doc`, which
  drives `LazyDocsCursor::advance(doc)` into it.
- `blocktree::FieldTerms::occurrences_for_doc` now calls it. A singleton term
  (`docFreq == 1`, pulsed into the term dictionary, no `.doc` bytes at all)
  keeps the old route, because there is no skip data for it to walk.

Tests: `crates/lucene-codecs/tests/postings_skip_fixture.rs` (F4) against real
Lucene, plus `crates/lucene-codecs/tests/postings_skip_pointers.rs` — every
document of a 700-document term across four field shapes (positions;
+payloads; +offsets; +both), a sampled walk over an 8 500-document term whose
level-1 entry the sample brackets, the singleton, and absent doc ids in the
middle and past the end. Each expectation is differential against
`FieldTerms::positions`, which reads no skip pointer at all.

Both halves were mutation-checked, at both skip levels:

| mutation | caught by |
|---|---|
| delete `skip_level1_to`'s `level0_pos = level1_pos` | the level-1 test (`doc 20612` decoded another document's occurrences) |
| delete `advance_shallow`'s `level0_pos.advance(skip)` | three of the four unit tests **and** the real-Lucene fixture |
| write the **level-0** `posBufferUpto` as a constant `0` | the same three |
| write the **level-1** `posBufferUpto` as a constant `0` | the level-1 test |
| ignore the **level-1** `posBufferUpto` on the read side | the real-Lucene fixture |

The last two rows are there because of the Tier-2 review; see below.

### F2 `[MISSING → fixed]` the write side emitted no pos/pay skip sub-fields, and refused the terms that would need them

**Java**: `flushDocBlock` writes the level-0 pair inside `if (writeFreqs)`,
immediately after the impacts run and before `numSkipBytes` is sampled;
`writeLevel1SkipData` writes the same into the level-1 entry's scratch buffer,
after the impacts and inside the `skip1EndFP` extent but outside
`numImpactBytes`.

**We did**: neither — and `validate_field` rejected any term with
`docFreq >= BLOCK_SIZE` while positions were indexed
(`Error::DocFreqTooLargeForPositions`), explicitly because `write_full_block`
did not emit them. So this port could not *write* a positions-indexing term
long enough to have skip data, which is why F1 had no test data of its own.

**Fixed**: `PosSkipWriter` holds Java's four running pointers
(`level0LastPosFP`/`level1LastPosFP` and the `.pay` twins, reset per term at
`startTerm`) and writes both records; `write_full_block` and
`write_level1_span` take it. The ceiling and its error variant are gone.

The one real design question: Lucene interleaves `.doc`/`.pos`/`.pay` as
documents arrive and *samples* `posOut.getFilePointer()`/`posBufferUpto` live
at each flush. This writer builds each file whole, so `.pos`/`.pay` are now
laid out **first** and `write_position_tail` returns a `PositionLayout` — the
`.pos`/`.pay` offset at every full-block boundary, plus a per-occurrence
`payloadByteUpto` prefix — from which `PosSkipWriter::sample(occ)`
reconstructs exactly what Lucene would have sampled. The reconstruction is
exact rather than approximate because the flush schedule is arithmetic: one
`.pos` block closes every 256 occurrences, doc-boundary-agnostic
(`addPosition`'s `posBufferUpto == BLOCK_SIZE`).

Tests: `postings_writer::tests::doc_freq_at_or_above_block_size_while_indexing_positions_round_trips`
(the shape that used to be an error), and all of `postings_skip_pointers.rs`,
whose expectations come from the reader that ignores skip data — so a
writer-side error cannot cancel against a reader-side one.

### F3 `[CORRECTNESS → fixed]` `lastPosBlockOffset` is load-bearing on the read side now, and nothing proved it

b5 F4 found this field being written as a constant `0`, which made real
Lucene's `refillPositions` treat a term's *first* position block as the vint
tail for every term with `totalTermFreq > 256`. b5 fixed the writer, and noted
that the defect had been invisible because **this port's own reader re-derived
the split from `total_term_freq`** and never read the field.

That re-derivation is exactly what a skip-driven walk cannot do: it jumps into
the middle of `.pos` with no running occurrence count, so the only thing
separating a `PForUtil` block from the vint tail is
`posStartFP + lastPosBlockOffset`. `last_pos_block_fp` now computes Java's
`reset` rule (`< BLOCK_SIZE` → `posStartFP`; `== BLOCK_SIZE` → no tail at all,
Java's `-1` sentinel becoming `None`; `> BLOCK_SIZE` → `posStartFP + offset`)
and `walk_document_occurrences` dispatches on it exactly as `refillPositions`
does.

**Fixed** in the sense the brief asked for: the field is now *used*, so it is
now *testable*. `a_wrong_last_pos_block_offset_is_visible_to_the_skip_driven_walk`
hands the reader a doctored `TermMetadata` with the offset zeroed — b5's exact
defect — and requires the walk to disagree with the whole-term reader;
`the_fixture_term_actually_carries_skip_data` asserts real Lucene records a
non-zero one. `blocktree::FieldTerms::term_metadata` was added so a test can
hand the reader a doctored metadata record without forging `.tim` bytes (which
its own footer would reject first, proving nothing).

`last_pos_block_fp_matches_lucene_reset` pins all three branches directly.

### F4 `[MISSING → fixed]` no fixture in the tree contained a single byte of this skip data

The differential floor for F1/F2, and it did not exist. `blocktree_index`'s
positions field ("pos") has `docFreq = 3` and `totalTermFreq = 4`: everything
is in the vint tail, no `.doc` full block exists, and therefore no level-0
header and no pos/pay sub-fields. Its "l1" field *does* have level-1 skip data
but indexes no positions, so the sub-fields are absent there too. No fixture
covered a full `.pos` `PForUtil` block against real Lucene either.

**Fixed**: `fixtures/src/GenPostingsSkip.java` →
`fixtures/data/postings_skip_index/`. One field
(`DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS`, with payloads), a term in **all
8 500 documents** with 25 500 occurrences — past `LEVEL1_NUM_DOCS`, so real
`Lucene104PostingsWriter` emits one level-1 entry, 33 level-0 headers and a
group-varint tail, each carrying the pos/pay pointers. Per-document
frequencies cycle 1..5, a period **coprime with 256**, so no `.pos` block
boundary lines up with a `.doc` block boundary and `posBufferUpto` is non-zero
in nearly every record — including the level-1 one, which is 253. That period
is load-bearing and the first revision of this fixture got it wrong; see the
Tier-2 review. Payload lengths cycle 0..2 so a block's payload byte run is not
uniform.

A second, sparser term (`gapterm`, 3 400 documents) shares the field. It is
not decoration: the dense term is in *every* document, so all 33 of its
level-0 blocks take Lucene's degenerate `docRange == BLOCK_SIZE` doc-delta
encoding, and without a second term the cross-engine ground truth would never
cover a skip-driven walk that has to bit-unpack a block, nor an
`advance(doc)` for a document the term does not contain.

Ground truth is Java's own `PostingsEnum.advance(doc)` + `nextPosition()` /
`startOffset()` / `endOffset()` / `getPayload()`, taken with a **fresh enum per
sampled document** so every sample is reached through the skip data rather
than by sequential iteration. 82 documents are sampled: either side of the
level-1 span end, either side of each of the 33 level-0 block ends, the first
document of the tail, first, last, and an irregular stride.

Read back by `crates/lucene-codecs/tests/postings_skip_fixture.rs`, four
tests: the structural assertions (the fixture really does carry a level-1
entry, a non-zero `lastPosBlockOffset` and a non-zero level-1
`posBufferUpto`), the skip-driven walk against Java's per-document ground
truth, the whole-term reader against the same ground truth, and the sparse
term's present-and-absent documents. The third is what makes the second
conclusive rather than circular, since the two readers share the per-block
wire decode but nothing of how they locate a document's occurrence window.

Generated with the checked-in 10.5.0 jars by running `GenPostingsSkip` alone
into `fixtures/data`; `gen-fixtures.sh` was **not** run in place, because it
regenerates every `IndexWriter` fixture with a fresh random segment ID and
`lucene-ffi/src/segment.rs` hardcodes one of them.

### F5 `[PERF → fixed]` and `[maintainability]` one wire decoder, not three

The new walk needs the same per-block decode `walk_wanted_occurrences` had
inline, and c15's stated reason for reimplementing `read_positions_for_docs`
on the shared walker was to have *one* wire decoder rather than two. Adding a
third would have undone that, and worse: `postings.rs` has produced eleven
panic-on-corrupt-input defects across five batches precisely where two copies
of the same decode disagree about a bound.

`refill_full_position_block` (`refillPositions` + `refillOffsetsOrPayloads`)
and `refill_last_position_block` (`refillLastPositionBlock`) are now the only
two places that decode a `.pos`/`.pay` block, and both walkers call them. The
tail decoder went from streaming per-occurrence to filling the block whole,
which is what Java does and costs nothing: a tail is at most 255 occurrences.

The one measurable change is that a block's payload byte run is now copied
into a reused buffer instead of borrowed from the mapped `.pay`. It costs
nothing in practice: the borrow only ever mattered to `PositionsOnly`, which
does not ask for payloads at all and so makes Java's `PForUtil.skip` +
`seek` call instead, and `FullOccurrences` copies each payload into its own
`Vec` regardless.

### F6 `[PERF → fixed, measured]` the residual, and what it is now

**Before** (c15's numbers, `benchmarks/.corpus/merged`, 5.2 M documents,
`body` indexed with offsets and payloads): highlighting one document of `t0`
(`docFreq` 4 997 130 / `totalTermFreq` 56 600 329) cost **18.3 ms** for the
first document and **25.3 ms** for the last, against 1.32 s for the whole-term
reader it replaced. c15 named the residual: the term's doc list (5.69 ms
measured on its own) plus the fused frequency-sum pass over five million
frequencies.

**After**, same corpus and bench, with a third arm added so the A/B/C is in
one build: `whole_term` is `FieldTerms::positions` (what the highlighter
called before c15), `one_doc_freq_sum` is c15's shape reproduced exactly
(`postings()` then `occurrences_for_docs` with the whole frequency list), and
`one_doc` is `FieldTerms::occurrences_for_doc` as it is now.

| term (`docFreq` / `totalTermFreq`) | document | `whole_term` | `one_doc_freq_sum` (c15) | `one_doc` (c20) |
|---|---|---|---|---|
| `t0` (4 997 130 / 56 600 329) | first | 2.7695 s | 34.406 ms | **1.5740 µs** |
| | middle | 2.4980 s | 33.207 ms | **7.7262 µs** |
| | last | 3.2666 s | 48.644 ms | **19.323 µs** |
| `t500` (4 711 / 4 713) | first | 323.61 µs | 8.0220 µs | **568.56 ns** |
| | middle | 208.79 µs | 6.7053 µs | **756.88 ns** |
| | last | 217.39 µs | 9.8432 µs | **985.72 ns** |
| `t999` (2 506 / 2 507) | first | 90.261 µs | 3.9212 µs | **591.69 ns** |
| | middle | 84.530 µs | 4.4179 µs | **1.0122 µs** |
| | last | 87.228 µs | 7.1353 µs | **2.2385 µs** |

The `one_doc` column was re-measured after the Tier-2 review's fixes (which
add a `check_wire_position` per level-0 header and one bounds check per call)
and is unchanged within the noise: `t0` first/middle/last come back at
1.194 / 9.205 / 15.675 µs against the 1.574 / 7.726 / 19.323 µs above, on a
differently-loaded machine.

**Read the absolute numbers with care and the ratios without**: this run
shared the machine with several other sweep agents (a `cargo llvm-cov` of two
crates among them), and every arm is inflated against c15's — `whole_term` on
`t0` measures 2.77 s here against c15's 1.32 s, and `one_doc_freq_sum` 34.4 ms
against c15's 18.3 ms, both roughly 2x. All three arms are in the same build
and interleaved by criterion, so the comparison between them is sound; the
comparison to c15's published absolutes is not.

**The residual, and its cause.** `t0`'s first document costs 1.57 µs and its
last 19.32 µs, and the difference is entirely `advance`'s **level-1 entry
walk**: `skipLevel1To` is a linear loop, so reaching document *n* reads
`n / LEVEL1_NUM_DOCS` level-1 entries, which is 0 for the first document and
610 for the last. `(19.32 − 1.57) / 610 ≈ 29 ns` per entry — a seek, six
varints and a `skipBytes` over the impacts run — and nothing else in the walk
scales with the document's position: the level-0 headers under the final span
are at most 32, exactly one 256-document block is decoded, and `.pos`/`.pay`
cost the one or two blocks holding the document's occurrences. `t500` and
`t999` (`docFreq` 4 711 and 2 506, below `LEVEL1_NUM_DOCS`) have no level-1
entries at all and are flat at 0.6–2.2 µs, which is the same statement from
the other side.

Java has the same asymptotics — `Lucene104` has two skip levels and
`skipLevel1To` walks them linearly — so this residual is not a divergence to
fix here. Getting past it would mean a skip *index* over the level-1 entries,
which is a format change.

**The doc list that is no longer read**, for scale: `postings()` on `t0` alone
is 5.69 ms (`docs_only_postings/t0_df4997130_freqs`, 5.45 ms for the
`DocsOnly` variant). Every `one_doc_freq_sum` figure above contains it and
every `one_doc` figure contains none of it.

**Phrase paths** (`phrase_positions` group, the two-term phrase `t0 t1`): this
batch does not change how they address `.pos` (F8), but it does rewrite the
block decode underneath them (F5), so they are re-run as a
no-regression check rather than as an improvement.

| `phrase_positions/t0_t1` | c15 | c20 |
|---|---|---|
| `all_positions_then_intersect` | 627.16 ms | 755.81 ms |
| `intersect_then_positions` | 493.99 ms | 542.64 ms |
| ratio | 1.27x | **1.39x** |

Both arms are inflated by the same shared machine, and the ratio between them
is unchanged-to-slightly-better, so the block-decode refactor costs the batch
path nothing. That is the answer the check was for: the payload byte run is
now copied into a reused buffer instead of borrowed from the mapped `.pay`,
and `PositionsOnly` — which is what this arm uses — never asks for payloads,
so it takes `PForUtil.skip` past them exactly as before.

### F7 `[PERF → fixed, measured]` level-1 impacts were decoded for every span *skipped*, and that became the residual

It was not the residual until this batch made it one. **b5's** F7 recorded
that `read_level1_entry` decoded — and allocated a `Vec` for — the impacts of
every span, including spans being skipped over, and judged it "bounded and
small: one level-1 entry per 8192 docs". That was right while `advance` was
not on the critical path of a highlight. With the
`.pos`/`.pay` pointers wired up it became the *dominant* cost: measured
mid-batch, before the impacts fix, `t0`'s middle document cost **37.3 µs** and
its last **175.2 µs** — against 7.7 µs and 19.3 µs after, on a machine that
was *more* loaded. `Level1Entry` now holds the run undecoded, exactly as
`FullBlockHeader::impact_bytes` already did, and `skip_level1_to` decodes it
only on the span it stops in (`skipLevel1To`'s
`needsImpacts && level1LastDocID >= target`). b5's F7 closed.

### F8 `[INTENTIONAL]` the batch walker keeps its frequency-sum addressing

`read_occurrences_for_docs`/`read_positions_for_docs` take a `wanted` set of
indices into the term's doc list and address `.pos` by a running sum over
`freqs`. They could be rebuilt on the skip path, and deliberately are not.

Their callers — phrase matching and the span paths — intersect the terms' doc
lists first, so by the time they ask for positions they have *already* decoded
the whole doc list the sum walks, and the wanted set is a large fraction of it
(c15 measured `t0 ∩ t1` as most of `t0`). The skip path would replace a pass
over an array already in cache with one `advance` per wanted document, each
re-walking `.doc`'s block headers — more work, not less, at that density. It
is the *single*-document shape where the asymmetry pays, which is exactly the
shape the skip data can address.

Recorded rather than fixed, with the density threshold named: a batch caller
whose `wanted` set is sparse *relative to `docFreq`* would benefit, and none
of this port's callers is.

### F9 `[INTENTIONAL]` `payloadByteUpto` is parsed and discarded, and recomputing it is identical

`read_pos_skip` reads the level-0/level-1 `payloadByteUpto` vint and drops it.
That is not an oversight: Java stores it (`seekPosData`'s
`payloadByteUpto = payUpto`) and then **overwrites it** from the landing
block's own decoded payload lengths the moment any occurrence is skipped
(`skipPositions`' `payloadByteUpto = sumOverRange(payloadLengthBuffer, 0,
toSkip)`). The only case where the stored value survives is `toSkip == 0`,
which implies `posBufferUpto == 0`, which implies `payloadByteUpto == 0` —
the writer resets both together at every block flush (`addPosition`'s
`posBufferUpto == BLOCK_SIZE` branch).

So recomputing is not an approximation of Java, it is the same number by a
route that does not let a file-derived value index a byte run. It is still
parsed, because the bytes are there and the fields after it are not
self-delimiting.

### Bounding every newly-trusted value

The class the brief called out, and the reason this file has produced eleven
defects across five batches. Every value this batch starts trusting comes off
disk, so:

| Value | Wire type | Bound |
|---|---|---|
| `pos_end_fp_delta` / `pay_end_fp_delta` | vlong | accumulated with `wrapping_add` into a `u64`; validated only where it is seeked, which reports EOF |
| `pos_buffer_upto` | one byte | `0..=255` by construction, so it can never index past a 256-entry block — stated in the field's doc rather than re-checked |
| `pay_buffer_upto` | vint | never used numerically (F9) |
| `PositionOrigin::skip` | derived: a byte + ≤256 frequencies | each frequency rejected if negative (`u64::try_from`), sum `wrapping_add`; checked against `total_term_freq` before it drives anything |
| the whole-block skip loop | derived | refuses to step past `lastPosBlockFP` (Java's `assert`), and each step is a `pfor_skip` that fails at EOF |
| the landing offset inside a block | derived | checked against the block's own decoded length, not used to index it |
| `freq` from `.doc` | `PForUtil` value | rejected if negative or greater than the term's `totalTermFreq` |
| a second refill after the vint tail | derived | rejected: the tail is the last block, so a frequency still wanting occurrences past it is a disagreement, not something to decode the footer for |
| `origin.pos_fp`/`pay_fp` as an address | `u64` | `usize::try_from(..).unwrap_or(usize::MAX)`, so a 32-bit target gets an out-of-range seek rather than a truncated in-range one |
| payload byte offsets within a block | derived | `checked_add` + `get(..)`, as before |

Tests: `a_skip_that_steps_past_the_vint_tail_is_rejected`,
`a_skip_landing_past_the_blocks_own_length_is_rejected`,
`a_frequency_longer_than_the_position_stream_is_rejected`,
`a_position_origin_past_the_end_of_the_pos_file_is_an_error`,
`position_origin_needs_a_freqs_cursor`, plus the honest control
`a_skip_driven_walk_starts_at_the_occurrence_the_origin_names` so none of the
rest can pass for the wrong reason.

The two block-level guards were run against the un-guarded code, and it is
worth being exact about what they actually buy, because it is not what the
usual guard in this file buys:

- **the landing-offset check** does not prevent a panic. `to_skip < BLOCK_SIZE`
  after the skip loop and the buffers are fixed 256-entry arrays, so the index
  is always in range; without the check the walk silently returns whatever
  stale value sits in the array (measured: `position 0`, from an untouched
  slot). It converts a **wrong answer** into an error, which is the worse of
  the two failures to ship.
- **the step-past-the-tail check** converts a confusing `unexpected end of
  input at offset 53` — the vint tail being decoded as a `PForUtil` block and
  running off the file — into a diagnostic that names the disagreement. Java
  asserts here for the same reason.

The `a_frequency_longer_than_the_position_stream_is_rejected` case is the one
that *is* a real hazard: without the second-refill guard the walk decodes the
`.pos` footer as a `PForUtil` block and emits a fabricated occurrence
(measured: `position 43`) before eventually running out of file.

### Verdict

Swept clean. c15's F10 and the matching `LEDGER.md` item are closed.

`postings.rs` is at **97.33%** line coverage (`cargo llvm-cov -p lucene-codecs
-p lucene-index`, the same two-crate run c15 used), against **97.85%** at c15
— held above the bar across a batch that added ~700 lines. The half-point is
the Tier-2 review's defensive arms; the two of them a test can reach honestly
now have one (`a_level0_header_whose_num_skip_bytes_disagrees_with_its_fields_is_rejected`,
`a_frequency_that_dwarfs_the_pos_file_is_rejected_before_the_walk`).
`postings_writer.rs` is at **99.55%** and `blocktree.rs` at **96.81%**;
the two-crate total is **97.74%**.

Two things this batch *covered* that c15 listed as gaps: the level-1 entry's
`.pos`/`.pay` sub-fields (which "need a `docFreq >= 8192` term on a
positions-indexing field" — the new fixture and `postings_skip_pointers.rs`'s
8 500-document term are exactly that), and `read_level1_entry`'s pos/pay
branch generally.

Of the lines still missed, all but a handful are error arms needing a
hand-built corruption no writer can produce, or accessors only `lucene-search`
calls. The two worth naming because they are *this batch's*:
`walk_document_occurrences`' empty-tail arm (`totalTermFreq % BLOCK_SIZE == 0`
with a `lastPosBlockOffset` that points at the boundary anyway — reachable
only through doctored term metadata, and the sibling arm one line above it,
which is the same class, *is* covered), and `position_origin`'s
negative-frequency rejection, which needs a `.doc` freq block with the high
bit set and no writer here emits one. Both are the uniformity the file's rule
demands rather than paths a test can reach honestly.

---

## `crates/lucene-codecs/src/postings_writer.rs`

Java: `Lucene104PostingsWriter.flushDocBlock` (`:392-495`),
`writeLevel1SkipData` (`:500-535`), `addPosition` (`:315-358`), `startTerm`
(`:240-250`).

| Rust | Java | Verdict |
|---|---|---|
| `PositionLayout` + `write_position_tail`'s return | *(no Java equivalent: Lucene samples live, this writer reconstructs)* | not-in-Java, exact (F2) |
| `PosSkipWriter::{new, add_block_docs, write_level0, write_level1}` | `level0LastPosFP`/`level1LastPosFP` + the two `if (writePositions)` regions | **added** (F2) |
| `write_full_block` | `flushDocBlock`'s `docBufferUpto == BLOCK_SIZE` branch | now writes the level-0 pos/pay region, and `numSkipBytes` is sampled after it, as Java does |
| `write_level1_span` | `writeLevel1SkipData` + its 32 `flushDocBlock` calls | now writes the level-1 pos/pay region into the scratch buffer between the impacts and `skip1EndFP`, and `numImpactBytes` is sampled before it, as Java does |
| `validate_field`'s `DocFreqTooLargeForPositions` | *(none — Java has no such limit)* | **removed** (F2) |
| `write_single_field`'s file order | *(Java interleaves)* | `.pos`/`.pay` now laid out before `.doc`, which points into them |

### Verdict

Swept clean for this batch's scope. The scope cuts b5 recorded (one `.tim`
leaf block per field, no zigzag/run-length compactness encodings, norm-1
impacts) are unchanged.

**Still open, and now more visible**: there is no cross-engine verifier for
the postings / term-dictionary write path at all (`verify-write-path.sh` says
so in its own header, filed as T3.1). Real Lucene has never read this port's
skip records — the evidence for them is this port's own two independent
readers agreeing with each other and with Java's *reader* on Java's bytes.
Carried over below.

---

## Gates

- `cargo fmt --all` — clean.
- `cargo clippy -p lucene-codecs --all-targets -- -D warnings` — clean.
- `cargo test -p lucene-codecs` — **1 285 passed, 0 failed**, including this
  batch's 18 new ones (9 unit, 4 in `postings_skip_pointers.rs`, 4 in
  `postings_skip_fixture.rs`, and 1 replacing the write-side test that used to
  assert the removed `docFreq` ceiling). Absolute totals move under this batch
  because other batches are adding tests to the same crate concurrently.
- `cargo test -p lucene-index` — **604 passed, 0 failed**.
- `cargo test -p lucene-search` — **1 020 passed, 0 failed**. This batch edits
  one doc comment in `highlighter.rs` and adds one arm to
  `benches/highlight_offsets.rs`; the highlighter's own behaviour change is
  entirely underneath it, in `FieldTerms::occurrences_for_doc`.
- `python3 scripts/check-parity.py` — ok.
- `scripts/verify-write-path.sh` — **21/21**, confirmed rather than assumed.
  It was 20/20 at c15; the 21st case arrived from another batch while this one
  ran. This batch adds none — the postings write path still has no verifier at
  all, which is the sharpest thing this batch leaves open (see the
  carry-over).
- `cargo llvm-cov -p lucene-codecs -p lucene-index --summary-only` —
  `postings.rs` **97.33%**, `postings_writer.rs` **99.55%**, `blocktree.rs`
  **96.81%**, two-crate total **97.74%**. `check_index.rs` is **93.75%**,
  below the bar and inherited (c15 recorded it at 89.77%; it is c19's item,
  not this batch's, and it has been climbing under that batch while this one
  ran).

**A note on the shared tree.** This batch ran alongside several others.
`lucene-index` and `lucene-search` were transiently un-buildable three times
while `merge.rs`, `vectors.rs` and `hnsw_vectors` were mid-refactor, and two
`index_writer` tests failed in one snapshot and had been deleted by their
owning batch by the next. Every gate above was re-run to green after those
settled; none of the breakage was in this batch's files, and none of it was
caused by it.

## Tier-2 review (`quality-reviewer`)

Run on this batch's scope, with the Java read alongside from the pinned tag
and the fixture's `.doc` bytes hand-decoded rather than taken on trust. It
independently confirmed the state machine at all six entry points,
`position_origin`'s equivalence to `posPendingCount` at the moment
`skipPositions` runs, the occurrence-for-occurrence walk, the write-side
framing (`numSkipBytes` / `blockLength` / `skip1EndFP` / `numImpactBytes`,
verified empirically against the fixture), and the `Level1Entry` impacts
change.

**One gating finding, and it was the one that mattered.** The fixture's
per-document frequency cycle was `1 + (d % 4)`, and `sum(1 + d % 4)` over the
first 8 192 documents is **20 480 = 80 × 256** — the level-1 span boundary
landed *exactly* on a `.pos` block boundary, so the level-1 record's
`posBufferUpto` was `0`. `postings_skip_pointers.rs` used the identical cycle,
so the port's own writer emitted `0` there too. Result: the level-1
`posBufferUpto` byte — on both the read and the write side, in the record this
batch singles out as the load-bearing one — was **indistinguishable from a
hardcoded zero** by every test the batch added, and the "coprime with 256"
claim in the test's own comment was false (4 is not coprime with 256; that is
the cause). Both generators now cycle 1..5, the fixture is regenerated
(level-1 `posBufferUpto` = 253), the manifest carries the value so the test can
assert the fixture stays non-degenerate, and the two mutations that were
previously invisible are now in the mutation table above.

The reviewer's suggested mechanical rule is worth keeping: for every wire
sub-field a batch newly starts trusting, assert the fixture contains a
*non-degenerate* value for it, not merely that the record exists. This batch
already did that for `lastPosBlockOffset` and not for `posBufferUpto`, which
is exactly the asymmetry that let it through.

Five advisories, all acted on:

- **`level0NumBytes` was parsed and discarded**, where `read_level1_entry`
  already cross-checks itself against `skip1EndFP`. It matters more now that
  the region it spans carries two variable-width sub-fields whose presence is
  decided by `FieldInfos` rather than by the stream, so a wrong `has_payloads`
  mis-frames the body and it decodes plausible garbage. Now checked with
  `check_wire_position`. This turned up four hand-built test headers writing
  the wrong value (the block *body*'s length, where Lucene writes the
  metadata region plus the two header fields) — fixed, and the helper that
  built most of them now measures the two regions separately.
- **Nothing bounded the occurrence count against the `.pos` bytes that exist.**
  `freq` is checked against `total_term_freq`, but `total_term_freq` is itself
  an unvalidated `.tim` vlong, so a corrupt segment where the two *agree*
  could grow ~256 `Position` records per byte of `.pos` before EOF — an
  allocation blow-up, and an allocation failure aborts. Now capped by
  `BLOCK_SIZE` occurrences per remaining `.pos` byte, which is the densest any
  block can be. The batch walker never had this exposure, because
  `wanted_ranges` requires the frequency list to sum to `total_term_freq`
  exactly.
- **`code >> 1` where Java has `code >>> 1`** in the vint tail's position and
  offset decode. `(delta << 1) | 1` is negative for a delta at or above 2^30,
  which `IndexWriter.MAX_POSITION` permits, and an arithmetic shift recovers
  the wrong delta. Pre-existing, but this function was rewritten here, so it
  is fixed in both the new decoder and the whole-term one — they are asserted
  equivalent, so they had to move together.
- **F9's stated reasoning was wrong even though its conclusion was right.**
  `payloadByteUpto` is not overwritten only when an occurrence is skipped; it
  is overwritten on *every* path after a seek, because `seekPosData` leaves
  `posBufferUpto == BLOCK_SIZE` and both refills assign `0`. The stronger
  statement is now in the code.
- **`PositionLayout::sample` mixed one panicking index with two checked ones.**
  Safe today (`validate_field` bounds it), but all three are checked now, with
  the invariant that bounds them named.

## Carry-over items raised by this batch

- [ ] **No cross-engine verifier for the postings write path** (T3.1,
      pre-existing). This batch adds a whole new wire region to what this port
      writes — the level-0/level-1 `.pos`/`.pay` skip records — and the only
      thing standing behind it is that two of this port's readers agree with
      each other. The natural shape is a `write_postings_skip_fixture` example
      plus a `VerifyPostingsSkip.java` that opens the six files with a
      hand-built `SegmentReadState` and drives `PostingsEnum.advance(doc)` +
      `nextPosition()`, which is precisely the API that reads them.
- [ ] **`IndexWriter` still cannot index positions**, so no full-segment
      fixture and no `CheckIndex` run has ever seen a `.pos`/`.pay` file this
      port wrote. `write_full_segment_fixture` uses `DocsAndFreqs`. Now that
      `postings_writer` has no positions-specific `docFreq` ceiling, the
      remaining blocker is `indexing_chain`/`segment_writer` (c17's files).
- [ ] **`postings.rs` is still on the arithmetic gate's TODO list**
      (`#[allow(clippy::arithmetic_side_effects)] // TODO(arith-audit)` in
      `lib.rs`). Everything this batch added is written to satisfy the gate,
      but the module as a whole has not been burned down, so nothing mechanical
      is checking it yet. c15's carry-over, unchanged.
