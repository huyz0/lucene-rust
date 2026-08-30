# M2 sweep — batch `b5-postings`

Files swept (every function in each):

- `crates/lucene-codecs/src/postings.rs`
- `crates/lucene-codecs/src/postings_writer.rs`

Java source of truth: `/home/tuong/work/lucene` @ 10.5.0
(`git log -1` = `091a987a14d`). Package confirmed by `ls`:
`org/apache/lucene/codecs/lucene104/` holds `Lucene104PostingsFormat`,
`Lucene104PostingsReader`, `Lucene104PostingsWriter`, `PostingsUtil`,
`ForUtil`, `PForUtil`, `PostingIndexInput`. There is **no**
`lucene104/blocktree` — the term dictionary lives in
`org/apache/lucene/codecs/lucene103/blocktree/` (swept in `b4`), and
`postings_writer.rs`'s `.tim`/`.tip`/`.tmd` half is compared against
`Lucene103BlockTreeTermsWriter` where it overlaps.

Also compared against: `codecs/{PushPostingsWriterBase,
CompetitiveImpactAccumulator,Impact,BlockTermState,PostingsReaderBase,
PostingsWriterBase}.java`, `index/{PostingsEnum,ImpactsEnum,Impacts,
ImpactsSource,TermState,FreqAndNormBuffer}.java`.

**Skip structure, confirmed by reading the source rather than assuming**:
Lucene104 has **no `MultiLevelSkipListReader`/`Writer` and no `skipOffset`**
term-metadata field. `codecs/MultiLevelSkipList{Reader,Writer}.java` still
exist in the tree but nothing in `lucene104` references them. Skip data is
inline: a **level-0** header immediately before every full 256-doc block
(`level0NumBytes` vlong, `docDelta` via `writeVInt15`, `blockLength` via
`writeVLong15`, then — only when the field has freqs — a vint-prefixed
impacts run and, only when it also has positions, `posEndFP` delta vlong +
`posBufferUpto` byte + optional `payEndFP` delta vlong + `payloadByteUpto`
vint), and a **level-1** entry before every span of exactly 32 full blocks
while ≥ 8192 docs remain (`docDelta` vint, span-length vlong, then — only
with freqs — a `skip1EndFP` `short`, a `numImpactBytes` `short`, the impact
bytes, and the same pos/pay sub-fields). **This port implements exactly that
shape, on both the read and the write side**, and this sweep found no
divergence in the layout itself.

## Summary

12 findings, F1-F12.

| Class | Count | Findings |
|---|---|---|
| `CORRECTNESS` | 4 | F1, F2, F3, F4 — all fixed with tests |
| `MISSING` | 0 | — (the Java methods with no Rust counterpart are all optimizations or JVM-lifecycle glue; see F9/F10) |
| `PERF` | 3 | F5 (fixed + benchmarked), F6, F7 (recorded open) |
| `INTENTIONAL` | 5 | F8, F9, F10, F11, F12 |

All `CORRECTNESS` findings are fixed with tests that were each confirmed to
fail against the pre-fix code.

---

## `crates/lucene-codecs/src/postings.rs`

Java counterpart: `lucene/core/src/java/org/apache/lucene/codecs/lucene104/
Lucene104PostingsReader.java` (+ `PostingsUtil.java` for the group-varint tail
block, `Lucene104PostingsFormat.java` for the constants).

### Method correspondence

| Rust | Java | Verdict |
|---|---|---|
| `DOC_CODEC`/`META_CODEC`/`POS_CODEC`/`PAY_CODEC`, `VERSION_START`/`VERSION_CURRENT` | `Lucene104PostingsFormat` constants | identical |
| `BLOCK_SIZE = 256` | `ForUtil.BLOCK_SIZE` / `Lucene104PostingsFormat.BLOCK_SIZE` | identical |
| `LEVEL1_NUM_DOCS = 32 * BLOCK_SIZE`, `LEVEL1_FACTOR = 32` | `LEVEL1_FACTOR`/`LEVEL1_NUM_DOCS`/`LEVEL1_MASK` | identical |
| `decode_term_metadata` | `decodeTerm` | identical (incl. the `absolute` FP reset, the singleton-zigzag branch, the `subsumes(POSITIONS)`/`subsumes(OFFSETS)||hasPayloads`/`totalTermFreq > BLOCK_SIZE` gates). Java's two `assert`s become hard errors — stricter, deliberately |
| `DocInput::open` | `Lucene104PostingsReader` ctor, `.doc` branch | identical (`checkIndexHeader` + `retrieveChecksum`) |
| `PosInput::open` / `PayInput::open` | same ctor, `.pos`/`.pay` branches | identical |
| `DocInput::read_postings` | `BlockPostingsEnum.reset` + `refillDocs` driven to exhaustion | divergent by design (eager whole-term materialization); wire handling identical |
| `DocInput::lazy_cursor` | `BlockPostingsEnum.reset` | identical state setup (`level1LastDocID = NO_MORE_DOCS` below 8192, `-1` + `level1DocEndFP = docStartFP` at/above) |
| `read_full_block_header` | `doMoveToNextLevel0Block`/`skipLevel0To`'s header read | identical field-for-field |
| `decode_full_block_body` | `refillFullBlock` | identical (PACKED / `bitsPerValue == 0` all-consecutive / UNARY bit set, then the `PForUtil` freq block) |
| `read_tail_block` | `refillRemainder` non-singleton branch + `PostingsUtil.readVIntBlock` | identical |
| `read_level1_entry` | `skipLevel1To`'s per-entry read | identical |
| `read_vint15` / `read_vlong15` | `readVInt15` / `readVLong15` | identical |
| `decode_impacts` / `decode_impacts_into` | `readImpacts` | identical |
| `singleton_postings` | `refillRemainder`'s `docFreq == 1` branch | identical |
| `LazyDocsCursor::{next_doc,advance,advance_shallow,refill,skip_level1_to}` | `nextDoc`/`advance`/`advanceShallow`+`doAdvanceShallow`+`skipLevel0To`/`refillDocs`/`skipLevel1To` | see F3, F4 |
| `LazyDocsCursor::{level0_impacts,level1_impacts,level0_last_doc_id,level1_last_doc_id}` | `Impacts.getImpacts(level)` / `getDocIdUpTo(level)` | divergent, see F9 |
| `LazyDocsCursor::freq` / `PostingsCursor::freq` | `freq()` | divergent: Java decodes the freq block lazily on first `freq()` (`freqFP`), this port always decodes it — see F6 |
| `read_positions` / `read_positions_flat` / `read_positions_for_docs` / `decode_position_streams` | `refillPositions` + `refillLastPositionBlock` + `refillOffsetsOrPayloads` + `nextPosition`/`startOffset`/`endOffset`/`getPayload` + `skipPositions` | batch-shaped rather than cursor-shaped, wire handling identical; `-1` offsets and empty payloads match `PostingsEnum`'s no-offsets/no-payload contract |
| `PostingsCursor` (whole type) | *not in Java* | Rust-only convenience over an already-decoded `Postings` |
| `find_impacts`, `find_next_geq`, `BlockScratch`, `PendingBlock` | `VectorUtil.findNextGEQ` / instance fields of `BlockPostingsEnum` | equivalent |
| `write_group_vints` | `GroupVIntUtil.writeGroupVInts` | identical (lives here as the write-side companion of `read_tail_block`) |

**Java methods with no Rust counterpart**: `init` (the `.tim`-embedded
postings header — handled by `blocktree.rs`), `newTermState`, `close`,
`toString`, `canReuse`, `prefetchPostings`, `checkIntegrity`, `cost`,
`intoBitSet`, `nextPostings`, `computeBufferEndBoundary`, `bufferIntoBitSet`,
`docIDRunEnd`, `sumOverRange`, `readLevel0PosData`/`seekPosData`,
`accumulatePendingPositions`/`accumulatePayloadAndOffsets`. All are covered by
F9/F10 below; none changes decoded output.

### Findings

**F1 [CORRECTNESS] `LazyDocsCursor::next_doc` answered from a stale block
after a shallow move.** *Java*: `nextDoc()` gates on
`if (doc == level0LastDocID || needsRefilling)` and, when `needsRefilling`,
calls `refillDocs()` — so after `advanceShallow(target)` it materializes the
shallow-positioned block and returns *its* first doc. *We*: `next_doc` only
checked `block_pos + 1 < block_len` and answered straight out of
`block_docs`, which after a shallow move still holds the **previous** block.
*Consequence*: a caller that shallow-advances and then calls `next_doc()` gets
a document behind the position the shallow move established, while
`level0_impacts()` already describes the *new* block — so a scoring loop could
score a document against another block's bound. `advance()` already carried the
`pending.is_none()` guard; `next_doc()` was missed when `advance_shallow` was
added. *Resolution*: **fixed** — `next_doc` now falls through to `advance` when
`pending.is_some()`, which refills and lands on the shallow block's first doc,
matching Java exactly. Test
`postings::tests::lazy_cursor_next_doc_after_advance_shallow_moves_to_the_shallow_block`
(verified to fail with `left: 1, right: 256` before the fix). Not reachable
from the current `lucene-search` callers, which always follow a shallow move
with `advance` — this is a latent public-API defect, not a live one.

**F2 [CORRECTNESS] `LazyDocsCursor::advance` panicked on a `.doc` whose block
body undershoots its own level-0 header.** A block's `docDelta` (header) and
its body's deltas are independent on the wire; nothing ties them together.
`advance_shallow` stops on a block whose header claims `last_doc_id >= target`,
then `advance` searches the refilled body for the first doc `>= target` and
indexed `block_docs[offset]` unconditionally. On a corrupt file `offset` can be
`BLOCK_SIZE`, which panics instead of surfacing a decode error — the same class
of defect the b-series already fixed in `read_positions`. *Resolution*:
**fixed** — returns `Error::Store(Corrupted(_))`. Test
`postings::tests::lazy_cursor_advance_rejects_a_block_body_that_undershoots_its_header`
(verified to panic with `index out of bounds: the len is 256 but the index is
256` before the fix).

**F5 [PERF] the tail block cost three heap allocations and two 256-entry
copies per term.** A term with `docFreq < BLOCK_SIZE` is *entirely* a tail
block, which is the overwhelmingly common shape, so this was a per-term cost on
the hot cursor path. `read_tail_block` allocated a `Vec<u64>` for the raw
group-varint values, and `LazyDocsCursor::advance` allocated two more `Vec`s
only to `copy_from_slice` them into the `block_docs`/`block_freqs` arrays it
already owned. Lucene's `refillRemainder` decodes straight into the
enumeration's long-lived `docBuffer`/`freqBuffer` and allocates nothing.
*Resolution*: **fixed** — `read_tail_block` now takes `&mut [i32]` destination
slices (the cursor passes `&mut self.block_docs[..count]` directly, the eager
path passes a resized sub-slice of its output `Vec`s) and reads group-varints
into a stack `[u64; BLOCK_SIZE]`, with a `count >= BLOCK_SIZE` guard so the
slice can never be short. Benchmarked with a new
`postings/lazy_cursor/{tail_block,full_blocks}` group in
`crates/lucene-codecs/benches/hot_paths.rs` (writes its own `.doc`/`.tim`/
`.tip`/`.tmd` through `postings_writer`, since no checked-in fixture has a term
long enough to time): a 200-doc tail-block walk went **1.27 µs → 1.00 µs
(−21%)** and a 2600-doc walk (10 full blocks + a 40-doc tail) **5.43 µs →
4.75 µs (−13%)**. The machine was noisy during the run (other sweep agents
compiling concurrently; criterion's confidence intervals were ±10%), so treat
the direction as established and the magnitude as approximate.

**F6 [PERF, recorded open] no `PostingsEnum` flags: freqs are always decoded.**
*Java*: `BlockPostingsEnum` separates `indexHasFreq` (the field's index
options) from `needsFreq` (the caller's `PostingsEnum.FREQS` flag). When the
field has freqs but the caller does not want them, `refillFullBlock` records
`freqFP` and calls `PForUtil.skip(docIn)` — reading one token byte and seeking
past the body — and `freq()` decodes lazily only if actually called. *We*:
`decode_full_block_body` always runs `pfor_decode` on the freq block, and
`read_tail_block` always reads the trailing freq-exception vints. *Consequence*:
a docs-only consumer (a filter clause, a conjunction leg whose score comes from
elsewhere, a merge that only needs doc IDs) pays a full 256-value `PForUtil`
unpack per block that Lucene skips. *Not fixed*: `read_postings`/`lazy_cursor`
have no flags parameter, so plumbing one means changing their signatures and
every call site in `blocktree.rs` and `lucene-search` — files owned by other
batches in this sweep. Recorded here rather than half-done; the contained shape
is a `needs_freq: bool` on `DocInput::lazy_cursor` plus a `for_util::pfor_skip`.

**F7 [PERF, recorded open] level-1 impacts are decoded even for spans being
skipped.** *Java*: `skipLevel1To` decodes the impacts run only when
`needsImpacts && level1LastDocID >= target`, and `skipBytes` past it otherwise.
*We*: `read_level1_entry` always calls `decode_impacts`, which also allocates a
fresh `Impacts` `Vec` per entry, and `skip_level1_to` then discards it for every
span it jumps. *Consequence*: bounded and small — one level-1 entry per 8192
docs, and its impacts run is a handful of vints — which is why this is recorded
rather than fixed. The level-0 path already does the right thing:
`FullBlockHeader::impact_bytes` is a borrow of the mapped file, decoded only
when a caller asks (that was a previous sweep's fix, and it is the shape this
one should copy if it ever matters).

**F9 [INTENTIONAL] `getImpacts` reports empty where Java synthesizes a
sentinel.** *Java*'s `Impacts.getImpacts(level)` returns `(freq=1, norm=1)`
when the field has no freqs, and `(freq=Integer.MAX_VALUE, norm=1)` for a level
whose extent is unknown (the tail block, or a level past the last one); its
`numLevels()`/`getDocIdUpTo(level)` complete the `ImpactsSource` contract. *We*
return an empty slice in exactly those states, and expose two concrete
accessors instead of the level-indexed trait. Both are safe: every consumer in
this port (`similarity::max_score_for_impacts`, the MAXSCORE loops in
`lucene-search/src/lib.rs`) treats an empty list as "no bound available,
cannot skip", which is the same conservative answer `Integer.MAX_VALUE`
produces. `level0_last_doc_id()`/`level1_last_doc_id()` carry the
`getDocIdUpTo` information under different names. Kept as-is: the sentinel only
exists to satisfy a Java interface this port does not have.

**F10 [INTENTIONAL] the bulk/vectorized and lifecycle methods are not ported.**
`intoBitSet`, `nextPostings`, `docIDRunEnd`, `computeBufferEndBoundary`,
`bufferIntoBitSet` are pure throughput optimizations for callers this port does
not have yet (`FixedBitSet`-collecting scorers, `DocAndFloatFeatureBuffer`
consumers, dense-run detection); `canReuse`/`newTermState`/`close`/`toString`
are JVM object-pooling and lifecycle glue that ownership makes unnecessary;
`prefetchPostings` needs an `IndexInput.prefetch`-equivalent madvise hook the
store layer does not expose; `cost()` is `docFreq`, which every caller here
already has. `checkIntegrity` (whole-file `checksumEntireFile`) is a
`CheckIndex`-time API — note `DocInput::open` *does* do the cheap
`retrieveChecksum` footer-structure check Java's constructor does, so the open
path is not weaker.

**F11 [INTENTIONAL] `advance(target <= docID())` is a documented no-op.** Java
forbids the call; this port defines it rather than leaving it to accident.
Unchanged by this sweep, already covered by tests.

---

## `crates/lucene-codecs/src/postings_writer.rs`

Java counterpart: `lucene/core/src/java/org/apache/lucene/codecs/lucene104/
Lucene104PostingsWriter.java` + `codecs/PushPostingsWriterBase.java`
(`setField`'s `writeFreqs`/`writePositions`/`writeOffsets`/`writePayloads`
derivation) + `codecs/CompetitiveImpactAccumulator.java`, plus
`lucene103/blocktree/Lucene103BlockTreeTermsWriter.java` for the `.tim`
block / `.tip` node / `.tmd` record this module also emits.

### Method correspondence

| Rust | Java | Verdict |
|---|---|---|
| `write_fields` / `write_single_field` | `Lucene104PostingsWriter` ctor + per-field/per-term drive loop + `close` | equivalent within scope (`.psm` maxima and file lengths match `close()` exactly) |
| `validate_field` | `startDoc`'s `CorruptIndexException` checks + `addPosition`'s asserts | stricter (Java asserts; we return typed errors) |
| `write_full_block` | `flushDocBlock`'s `docBufferUpto == BLOCK_SIZE` branch | see F3 |
| `write_level1_span` | `writeLevel1SkipData` + its 32 `flushDocBlock` calls | identical field-for-field (see F12 for impacts) |
| `write_tail_block` | `flushDocBlock`'s `docBufferUpto < BLOCK_SIZE` branch + `PostingsUtil.writeVIntBlock` | identical |
| `write_position_tail` | `addPosition`'s `posBufferUpto == BLOCK_SIZE` flush + `finishTerm`'s vint tail | identical, incl. payload-before-offset order and the `-1` "force the first length" seeds |
| `write_full_position_block` / `write_full_offset_block` / `write_full_payload_length_block` | `addPosition`'s three `pforUtil.encode` calls + the payload byte run | identical |
| `write_term_metadata` | `encodeTerm` | see F4 (and F8 for the zigzag branch) |
| `write_vint15` / `write_vlong15` | `writeVInt15` / `writeVLong15` | identical |
| `PostingsMaxima` | `maxNumImpactsAtLevel0`/`maxImpactNumBytesAtLevel0`/`…Level1` | identical accumulation |
| `write_tim_block` / `write_leaf_node` | `Lucene103BlockTreeTermsWriter.writeBlock` / `TrieBuilder` | scope-limited (single leaf block per field), see F8 |
| singleton pulsing in `write_fields` | `finishTerm`'s `docFreq == 1` branch | identical |
| *no counterpart* | `startTerm(NumericDocValues norms)`, `startDoc`'s norm lookup, `CompetitiveImpactAccumulator` | see F12 |

**Java methods with no Rust counterpart**: `startTerm`/`startDoc`/`finishDoc`/
`addPosition` as *incremental push callbacks* — this writer takes a fully
materialized `FieldPostingsInput` and does the same work in batch form, so the
push API has no shape to port. `writeImpacts` is subsumed by F12.

### Findings

**F3 [CORRECTNESS] the doc-delta encoding choice compared against the wrong
quantity — and the wrong direction was pinned by a test.** *Java*
(`Lucene104PostingsWriter.java:437-441`):

```java
int numBitSetLongs = FixedBitSet.bits2words(docRange);
int numBitsNextBitsPerValue = Math.min(Integer.SIZE, bitsPerValue + 1) * BLOCK_SIZE;
if (docRange == BLOCK_SIZE) { writeByte(0); }
else if (numBitsNextBitsPerValue <= (numBitSetLongs * Long.SIZE)) { /* packed FOR */ }
else { /* bit set */ }
```

*We* compared `num_bits_next_bits_per_value <= doc_range`, with a comment
asserting that this is what Lucene does. It is not: `doc_range <=
num_bit_set_longs * 64` always, so ours was a *stricter* packed condition and
picked the bit set for every block in
`doc_range < num_bits_next <= ceil(doc_range/64)*64` where Lucene picks packed
FOR. *Consequence*: not wrong bytes — both encodings are legal and both
readers take either — but the **larger** of the two, and the slower one for
`nextDoc`. The worked example is the existing test's own construction: 208
deltas of 3 and 48 of 2 (`docRange == 720`, `bitsPerValue == 2`) is 512 bits of
packed body against a 12-long / 768-bit bit set, and we wrote the 768.
`full_block_encoding_choice_matches_lucene_in_the_disputed_band` asserted the
wrong token (`-12`), with a doc comment stating Lucene's condition backwards —
a previous slice "fixed" this in the wrong direction. *Resolution*: **fixed**
— the comparison is now `num_bits_next_bits_per_value <= num_bit_set_longs *
64`, the test asserts `2` and its doc comment quotes the Java line. The two
sibling tests that pin the other outcomes (`full_block_dense_picks_bitset_token`
at `docRange == 257`: `768 > 5*64 == 320` → bit set;
`full_block_irregular_picks_plain_packed_token`: `2048 <= 202*64` → packed)
were already correct under both conditions and are unchanged, so they now
bracket the fixed comparison from both sides. Java's `assert numBitSetLongs <=
BLOCK_SIZE / 2` still holds: the new condition only *narrows* when the bit set
is chosen.

**F4 [CORRECTNESS] `lastPosBlockOffset` was written as a constant `0`.**
*Java* (`finishTerm`): when `totalTermFreq > BLOCK_SIZE`, records
`posOut.getFilePointer() - posStartFP` — sampled **before** the vint tail is
written, so it is the byte length of the term's full `PForUtil` position
blocks, i.e. where the vint tail begins. `Lucene104PostingsReader.reset` turns
it back into `lastPosBlockFP = posStartFP + lastPosBlockOffset`, and
`refillPositions` switches from `pforUtil.decode` to `refillLastPositionBlock`
the instant `posIn.getFilePointer()` reaches it. *We* wrote `0`, on the
reasoning that this port's own `read_positions` re-derives the block/tail split
from `total_term_freq` and never reads the field. *Consequence*: real Lucene's
`lastPosBlockFP` equals `posStartFP`, which is true at the term's *first*
position block — so Lucene decodes a `PForUtil` block as a vint tail and
produces garbage positions/offsets/payloads for **every term with
`totalTermFreq > 256`**. Invisible to every test in the repo (this port's
reader ignores the field, and `IndexWriter` cannot write positions yet), and
invisible to `scripts/verify-write-path.sh` for the same reason. *Resolution*:
**fixed** — `write_position_tail` now returns the offset, sampled at exactly
the point Java samples it; `write_fields` threads a per-term
`last_pos_block_offset` vector through `write_tim_block` into
`write_term_metadata`. Test
`postings_writer::tests::last_pos_block_offset_locates_the_vint_position_tail`
writes a `totalTermFreq == 300` term end to end, walks the real `.tim` leaf
block down to its metadata region, decodes it through the *unmodified*
`postings::decode_term_metadata`, and then proves the offset by reading the 44
vint-tail position deltas out of `.pos` at `posStartFP + offset` and asserting
they end exactly where the `.pos` footer begins.

**F8 [INTENTIONAL] two compactness-only encodings are not emitted.**
`encodeTerm`'s zigzag-singleton-delta branch (for runs of `docFreq == 1` terms
sharing a `docStartFP`) and `StatsWriter`'s singleton run-length encoding
(`((singletonCount - 1) << 1) | 1`) are both strictly smaller alternatives to
the plain forms this writer emits. Both readers accept the plain form — real
Lucene's `decodeTerm` takes the bit-clear branch and its `StatsReader` takes
the even code — so this is file size, not correctness. Related and already
recorded in `docs/parity.md`: one `.tim` leaf block per field (no floor
blocks, no non-leaf sub-block entries), which is the larger scope cut.

**F12 [INTENTIONAL] `CompetitiveImpactAccumulator` is not ported, and for this
writer's inputs it does not need to be.** This writer emits exactly one impact
per level-0 block and one per level-1 span: `(maxFreq, norm = 1)`. That is not
an approximation of Java's output — it *is* Java's output for a field with no
norms. `Lucene104PostingsWriter.startDoc` passes `norm = 1L` whenever
`fieldHasNorms == false`, and
`CompetitiveImpactAccumulator.getCompetitiveFreqNormPairs` with a constant norm
keeps only the single highest-freq entry (`maxFreqs[1] = maxFreq`, everything
else 0, and the `maxFreq > maxFreqForLowerNorms` loop emits one pair). The
level-1 entry is `addAll` of the span's level-0 accumulators, which under the
same constant norm is the span-wide max freq — which is what
`write_level1_span` computes. The genuine gap is that
`FieldPostingsInput` carries no norms at all, so a field that *does* index
norms gets a bound computed against norm 1 rather than its real per-doc norms.
That bound is still sound (norm 1 is the shortest field length, hence the
highest score), so it costs pruning opportunities and never drops a hit.
Closing it means adding a norms input to this writer's public API and to
`lucene-index`'s call sites — out of this batch's files. Recorded, unchanged.

---

## Verdict

### `crates/lucene-codecs/src/postings.rs`

Swept clean on the wire format: every field of the level-0 header, the level-1
entry, all three doc-delta encodings, the group-varint tail, the impacts run,
the `.pos`/`.pay` full blocks and vint tail, and `decodeTerm`'s full flag
matrix match `Lucene104PostingsReader` field for field and gate for gate. Two
cursor state-machine defects fixed (F1, F2) and one allocation-shape
regression fixed and benchmarked (F5). Open by choice: F6 (no
`PostingsEnum`-flags plumbing, so freqs are always decoded — needs signature
changes in files owned by other batches) and F7 (level-1 impacts decoded for
skipped spans — bounded, one per 8192 docs).

### `crates/lucene-codecs/src/postings_writer.rs`

Two real wire defects fixed: the doc-delta encoding heuristic (F3, which also
had a test pinning the wrong behaviour and a doc comment stating Java's
condition backwards) and `lastPosBlockOffset` (F4, which would have corrupted
positions for any `totalTermFreq > 256` term read by real Lucene, and which no
existing test or cross-engine check could see). The remaining divergences are
the already-documented scope cuts — one `.tim` leaf block per field, no
run-length/zigzag compactness encodings, no norms input and therefore
norm-1 impacts (F8, F12) — none of which produce bytes real Lucene rejects.
