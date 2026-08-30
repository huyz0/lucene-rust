# c31 — the arithmetic gate, burned down across the last five `lucene-codecs` modules

Follow-up to c24 (14 modules) and c27 (7 modules), which between them left
**five** carrying a `#[allow(clippy::arithmetic_side_effects)] //
TODO(arith-audit)` marker: `fst`, `hnsw`, `hnsw_vectors`, `postings_writer`,
`vectors`. This batch audits all five. `docs/arithmetic-gate.md`'s table goes
**5 → none**: `lucene-codecs` is now "on, fully audited".

Java read from **`/home/tuong/work/lucene-10.5.0`**, the pinned tag.

## Starting state

A previous attempt at this batch had deleted the five markers from `lib.rs`
without finishing the audit, so the crate's gate was red:
`cargo clippy -p lucene-codecs --all-targets` reported **197
`arithmetic_side_effects` errors** (`fst.rs` 105, `postings_writer.rs` 60,
`vectors.rs` 32) plus **7 `#[allow]`s in `hnsw.rs` with no `// ARITH:` proof**
(`scripts/check-arith-allows.py` exit 1). It also left a stray
`examples/c31_ab_bench.rs` behind, now deleted.

`hnsw.rs` and `hnsw_vectors.rs` were already lint-clean and, on inspection,
already carefully audited for the non-lint half as well — see
[INTENTIONAL-1](#intentional-1-two-modules-were-already-clean).

## Burn-down

| | count |
|---|---|
| modules carrying the marker after c27 | 5 |
| **audited this batch** | **5** |
| lint sites resolved (lib, non-test) | **174** |
| lint sites opted out at a test-module boundary | 23 |
| unjustified `#[allow]`s given proofs | 7 |
| **remaining marked in `lucene-codecs`** | **0** |

**Findings: 18 `CORRECTNESS`/`MISSING`, all fixed; 5 defensive hardening;
2 `PERF`; 2 `INTENTIONAL`.** Of the 18, **thirteen came from the half the lint does
not cover** — allocation, indexing, `debug_assert`s on file-derived values and
an unbounded descent — and those include the batch's two verified process
kills.

| what it does when it fires | count |
|---|---|
| **OOM kill** (unbounded FST descent; worse than an abort — no timeout fires) | 1 verified (SIGKILL under the container's 8 GiB cap) |
| **abort** (allocation sized by a number just read off disk) | 3 |
| **silent wrong answer** in a release build | 4 |
| **panic** (debug arithmetic, or an index/slice in both) | the rest |

| Rust file | Java counterpart (10.5.0) | lib sites |
|---|---|---|
| `fst.rs` | `util/fst/{FST,FSTEnum,BytesRefFSTEnum,IntsRefFSTEnum,BitTableUtil,ReverseBytesReader,ByteSequenceOutputs,PositiveIntOutputs,PairOutputs,Util}.java` | 98 |
| `postings_writer.rs` | `codecs/lucene104/Lucene104PostingsWriter.java`, `codecs/lucene103/blocktree/Lucene103BlockTreeTermsWriter.java` | 47 |
| `vectors.rs` | `codecs/lucene99/Lucene99FlatVectors{Reader,Writer}.java`, `codecs/lucene95/{OffHeapFloatVectorValues,OffHeapByteVectorValues,OrdToDocDISIReaderConfiguration}.java`, `util/VectorUtil.java` | 29 |
| `hnsw.rs` | `util/hnsw/{HnswGraph,HnswGraphBuilder,HnswGraphSearcher,OnHeapHnswGraph,NeighborArray,NeighborQueue,MergingHnswGraphBuilder,InitializedHnswGraphBuilder}.java` | 0 (7 proofs) |
| `hnsw_vectors.rs` | `codecs/lucene99/Lucene99HnswVectors{Reader,Writer}.java` | 0 |

---

## `fst.rs`

Java: `org.apache.lucene.util.fst.*`. 98 lint sites: 21 resolved with a
`checked_*`/`saturating_*`/typed rejection or a restructure, 77 with a
tightly-scoped `#[allow]` carrying an `// ARITH:` proof.

### F1 `[CORRECTNESS]` `Fst::read` — `numBytes` sized the body `Vec` unbounded (**abort**)

`FSTMetadata`'s `numBytes` is a `vlong`, validated non-negative and nothing
else, and it went straight into `vec![0u8; meta.num_bytes as usize]`. A single
flipped byte in that field reserves up to 2^63 bytes: the allocator aborts, and
`catch_unwind` cannot intercept an allocation failure, so the embedding JVM dies
with it. `read_borrowed` already bounded the same field through
`SliceInput::slice`; the owning path did not.

Fixed by bounding it against `input.len() - input.position()` — the bytes
actually left in the file — before the reservation. Test:
`read_rejects_a_body_length_no_allocation_could_hold`.
`read_rejects_truncated_body` now asserts `Error::Corrupt` rather than the
`Error::Store` EOF that used to arrive *after* the allocation.

### F2 `[CORRECTNESS]` the accepts-empty output length sized a second unbounded `Vec` (**abort**)

Same shape one field earlier: `read_fst_metadata_prefix`'s `vint` empty-output
length reached `vec![0u8; num_bytes]` with only a sign check in front of it, so
a corrupt flag byte plus a large length reserves 2 GB. Bounded against the same
remaining-bytes figure. Test:
`read_rejects_an_empty_output_length_no_allocation_could_hold`.

### F3 `[CORRECTNESS]` `BytesReader::read_output` — every arc's output length sized a `Vec` unbounded (**abort**)

The sharpest of the three, because it is per *arc* rather than per file.
`ByteSequenceOutputs.read`'s `int len = in.readVInt()` was ported as
`vec![0u8; len as usize]` with no check at all: a **negative** length
sign-extends through `as usize` to ~2^64 and aborts immediately, and a large
positive one reserves 2 GB. The `read_bytes` that would have failed runs
*after* the allocation.

Fixed with a `checked_output_len` helper shared by `read_output` and
`skip_output`. The bound is exact rather than arbitrary: the reverse cursor
consumes an output's bytes *downward* from the current position, so a valid
length is at most `pos + 1`. Test:
`arc_output_length_is_bounded_by_the_body_below_the_cursor`.

### F4 `[CORRECTNESS]` a cyclic arc target made the enum descend until the OOM killer stopped it

`FSTEnum`'s `pushFirst`/`pushLast`/`doSeekExact` descend from an arc into its
target node in an unbounded loop. An FST is a DAG, so Java's loop terminates;
a corrupt body whose targets form a cycle makes it grow `arcs`, `outputs` and
`labels` one level per step — and `outputs` accumulates a longer byte string at
every level, so the growth is quadratic. This is **worse than an abort**:
nothing catches it, no timeout fires, and the process dies to the OOM killer.

Found by the new byte-flip sweep, which took a **SIGKILL** from a single
flipped target byte.

Fixed with `FstEnum::check_descent`: `FSTCompiler` writes nodes bottom-up into
a forward-growing byte store, so a node's address is always greater than every
address its arcs point at. A target that does not strictly precede its own node
is a cycle, and rejecting it terminates the descent immediately. The bound
cannot reject a real file — every FST fixture in `fixtures/data/fst*` (all four
node encodings, multi-level tries and array roots alike) walks fully with it in
place. Test:
`a_cyclic_arc_target_is_rejected_rather_than_descended_forever` (SIGKILLs
against the unfixed code, in a release build).

### F5 `[CORRECTNESS]` `(low + high) >> 1` where Java has `>>>`, twice

`FST.findTargetArc:1064` and `Util.binarySearch:913` are both
`int mid = (low + high) >>> 1;` — the unsigned shift is load-bearing because
`numArcs` comes off disk unbounded, so `low + high` overflows an `int`. Ported
as a signed `>>`, that yields a **negative midpoint** and binary-searches the
wrong side of the arc array (release), or panics on the overflowing add
(debug). This is the shape `docs/arithmetic-gate.md` names, and grepping the
Java for `>>>` found both occurrences in the package.

Both are now `(low.wrapping_add(high) as u32 >> 1) as i32`, which is Java's
shift bit for bit.

### F6 `[MISSING]` `read_vint`/`read_vlong` had no length bound

`DataInput.readVInt` reads at most five bytes and then throws
`"Invalid vInt detected (too many bits)"`; `readVLong` likewise. `fst.rs`'s own
reverse `BytesReader` re-implemented the algorithm without that bound (the
crate's `lucene_store::data_input` version has it), so a corrupt run of
continuation bytes returns a value built from bits Java rejects outright — and
the `shift` accumulator itself overflows after ~3·10^8 bytes. Both now carry
the same `shift > 28` / `shift >= 64` rejection. Test:
`a_vint_longer_than_five_bytes_is_rejected_not_silently_truncated`.

### F7 `[CORRECTNESS]` `labels_to_bytes` narrowed a 260 to a 4

`Fst::iter`'s guard means every *stored* label of a `BYTE1` FST is a byte — but
a direct-addressing or continuous node stores no labels at all: `read_arc`
derives each as `firstLabel + arcIdx`. A corrupt label range makes that exceed
255 on a `BYTE1` FST, and `labels_to_bytes` was a `debug_assert!` plus an
`as u8`: a panic in debug and, in release, a **plausible wrong key** (260
truncates to 4). Found by the byte-flip sweep on a builder-produced FST.

Fixed at the two places the label is formed rather than at the narrowing —
which is also why the narrowing stays free (see PERF-1):

- `check_label_range` bounds `firstLabel + numArcs` by the alphabet the
  `INPUT_TYPE` declares (256 for `BYTE1`, 65 536 for `BYTE2`). A real writer
  cannot emit a wider range: there are no such labels to put in it.
- `bit_table_next_bit_set` refuses an index past `numArcs`. The presence bit
  table is rounded up to whole bytes, so its last byte carries up to seven bits
  past the label range; `FSTCompiler` writes those as zero, and a set one would
  otherwise be returned as an `arcIdx` outside `0..numArcs`.

Tests: `a_label_range_outside_the_input_types_alphabet_is_rejected` (fails
against the unfixed code), `a_presence_bit_set_past_the_label_range_is_not_returned_as_an_arc`.

### F8 `[CORRECTNESS]` `read_label`'s `BYTE4` branch accepted a negative label

`FST.readLabel`'s `BYTE4` branch is a bare `readVInt()`, which decodes to any
`i32`. Every label a writer emits is non-negative (`END_LABEL == -1` is
synthetic and never stored), and a negative one makes both
`targetLabel - firstLabel` and `firstLabel + arcIdx` unbounded subtractions —
they are exactly the two expressions the batch's `// ARITH:` proofs rest on.
Now rejected. Test: `a_negative_byte4_label_is_rejected` (fails against the
unfixed code).

### F9 `[CORRECTNESS]` a node header with `numArcs <= 0` or `bytesPerArc <= 0`

Both are plain `vint`s and `FST.java` trusts both. `numArcs <= 0` makes
`numArcs - 1` (binary search's `high`) and `numArcs - 2`
(`readLastTargetArc`'s `arcIdx`) underflow, and `bytesPerArc == 0` collapses
every slot address onto one arc. `FSTCompiler` emits neither —
`ARCS_FOR_BINARY_SEARCH` needs at least `FIXED_LENGTH_ARC_SHALLOW_NUM_ARCS`
arcs to be chosen at all — so rejecting them cannot reject a real file, and it
is the single bound that makes the downstream proofs hold. Centralised in
`read_array_node_header`, used by all five sites that read such a header.
Test: `an_array_node_header_with_no_arcs_or_zero_width_slots_is_rejected`
(fails against the unfixed code).

### F10 `[CORRECTNESS]` `Outputs::decode` panicked on a payload off disk

`read_vint_forward`/`read_vlong_forward` indexed their slice directly
(`bytes[idx]`) and shifted by an unbounded amount; `PairOutputs::decode` sliced
`&bytes[header_len..header_len + first_len]` with `first_len` a `vint` off
disk (negative → ~2^64 through `as usize`; large → past the end). All three are
panics, and they are reachable: `suggest.rs` calls
`PositiveIntOutputs::decode` on the output bytes `Fst::get` walked out of a
real FST body.

The trait's doc comment claimed decode is "only ever called on bytes this same
`Outputs` impl produced" — which is false the moment those bytes come off disk.
`decode` now returns `Result`, both forward readers are bounded and use `get`,
and `PairOutputs` forms its split point with `checked_add` + `get`. Test:
`typed_output_decoders_reject_a_corrupt_payload_instead_of_panicking`.

### F11 `[CORRECTNESS]` `decode_ints_key` silently dropped a key's tail

`debug_assert_eq!(bytes.len() % 4, 0)` plus a `chunks_exact(4)`: a debug panic,
and in release a **shorter, entirely plausible key**. The length is not a
caller's, it is assembled from arc labels walked out of an FST — enumerating a
`BYTE1` FST through `iter_ints()` produces exactly such a length, as the sweep
demonstrated. Now a typed error. Test: extended
`encode_decode_ints_key_round_trips`.

### F12 `[CORRECTNESS]` eight `debug_assert`s on file-derived state

Java states each of these as an `assert`, i.e. a check that is off in
production. Ported as `debug_assert!` they were worse than that: a debug build
panics and a release build walks on with a nonsense arc, so the two *disagree*
on corrupt input and neither rejects it. Converted to typed
`Error::Corrupt` through a shared `corrupt_fst` constructor:
`readLastTargetArc`'s `follow.isFinal()` and its `arc.isLast()`
postcondition, `readNextArcLabel`'s `!arc.isLast()`, `readNextRealArc`'s
`0 <= arcIdx < numArcs`, `findNextFloorArcBinarySearch`'s `idx != -1`,
`seekCeilDirectAddressing`'s `ceilIndex != -1`, and both
`findNextFloorArc{DirectAddressing,Continuous}`'s `targetIndex >= 0`. The
byte-flip sweep reaches five of them from a single flipped byte.

An earlier draft of this section also listed `readNextRealArc`'s
`nextIndex != -1`. **It was not converted** -- see F20, which the Tier-2 review
found by checking the claim against the tree rather than against this report.

### F13 `[CORRECTNESS]` `num_presence_bytes` overflowed its own argument

`FST.getNumPresenceBytes` is `(labelRange + 7) >> 3` on an `int`, and
`labelRange` is `numArcs` off disk — within 7 of `Integer.MAX_VALUE` the add
overflows. Widened to `i64` for the one add, which is exact for every `i32`
input and still fits back in an `i32`.

### F13b `[CORRECTNESS]` the bit-table popcount accumulator was one short

`BitTableUtil.countBits`/`countBitsUpTo` were ported with an `i32`
accumulator. `numArcs` is an unbounded `vint`, so `getNumPresenceBytes` reaches
2^28 and eight bits per byte reaches **2^31 — exactly one past `i32::MAX`**.
Caught by re-checking this batch's own `// ARITH:` proof against its boundary
value, which is the failure mode three prior Tier-2 reviews found: the first
draft of the proof bounded `num_arcs` by the `INPUT_TYPE`'s alphabet, which
`check_label_range` does *not* do for `BYTE4` (its limit there is `i32::MAX`).
Both accumulators are now `i64` with an `i32::try_from` on the way out.

### F20 `[CORRECTNESS]` `bit_table_next_bit_set`'s `-1` reached `read_arc` as an `arcIdx`

**The hole F7 claimed to close, left open at one of its two call sites.** F7
bounded the *upper* end of `bit_table_next_bit_set`'s result (`index >=
num_arcs` -> `-1`) on the grounds that an `arcIdx` outside `0..numArcs` must
not reach `read_arc` -- but `-1` is outside that range too, and
`read_next_real_arc` passed it straight through.
`read_arc_by_direct_addressing` then set `arc_idx = -1` and `read_arc` derived
`first_label + arc_idx = first_label - 1`: an arc one label *below* the range
the node declared, and for `first_label == 0` exactly `END_LABEL`, a spurious
acceptance of the empty continuation. The sibling call site in
`read_next_arc_label` already had the check; this report claimed both did.

Reachable from `read_first_real_target_arc` with a **single bit flip** in a
direct-addressing node's presence byte: an all-zero table makes
`next_bit_set(-1)` return `-1`, `presence_index` goes `-1 -> 0`, and the slot
address is therefore the perfectly valid `pos_arcs_start`.

Three reasons the byte-flip sweep could not find it, now written up as a
standing rule in `docs/arithmetic-gate.md` ("Check an out-of-domain sentinel at
every call site"):

1. The sweep asserts "a typed error or a clean decode", and a plausible wrong
   label is a clean decode. It ran 40 136 flips over this code and passed.
2. `first_label` is an ASCII letter in every FST fixture, so `first_label - 1`
   is an ordinary label there; only `first_label == 0` makes it `END_LABEL`.
3. The bad value produces a *valid* address, so the reader's own range check
   accepts it. "Caught downstream" was an assumption, not a fact.

Fixed by mirroring `read_next_arc_label`'s guard. Test:
`an_all_zero_presence_table_does_not_decode_an_arc_below_the_label_range`,
built at `first_label = 0` -- it fails against the unfixed code in a **release**
build, i.e. on the wrong-answer path rather than a debug assert.

### F21 `[CORRECTNESS]` `countBits - 1` was the same sentinel, one function over

`read_last_arc_by_direct_addressing` computes Java's
`BitTable.countBits(arc, in) - 1`, which is `-1` for an all-zero table. This
report's first draft justified passing it on with "the resulting slot address
is then rejected by `read_byte`'s range check" -- **false**: a `-1`
`presence_index` addresses `pos_arcs_start + bytes_per_arc`, *above* the arc
array's start and normally well inside the body, so `read_byte` accepts it and
decodes a garbage arc.

Now rejected. `FSTCompiler.writeNodeForDirectAddressing` only emits a
direct-addressing node for a run of arcs it has already written, and every one
of them sets its presence bit, so a node with no set bit cannot come from a
real writer. Test:
`an_all_zero_presence_table_is_rejected_by_the_last_arc_path_too`, also failing
against the unfixed code in release.

### F22 `[CORRECTNESS]` two proofs cited a Java exception that is not in the pinned tree

`read_vint`/`read_vlong`'s bounds were justified as "`DataInput.readVInt`'s own
`Invalid vInt detected (too many bits)`". **That exception does not exist in
10.5.0** -- `DataInput.readVInt` there is a bare unchecked `for` loop; the
message is a later addition and `grep` finds it only in `CHANGES.txt`. It is
`PROTOCOL.md`'s `main`-versus-pinned trap, in the evidence rather than in the
code.

The bounds are still right, on a better argument: the **writer** is what
constrains a real file. `DataOutput.writeVInt` emits "between one and five
bytes" (its loop is `i >>>= 7` on an `int`) and `DataOutput.writeVLong`
"between one and nine bytes" while rejecting negatives outright
(`IllegalArgumentException("cannot write negative vLong")`). So the `vint`
bound matches the writer exactly and the `vlong` bound is one byte looser --
which is why neither can reject a real file. Both comments now say that, and
note that they are deliberately stricter than 10.5.0's own reader.

The same false citation sat in `terms_dict.rs` (a c27-era module); found by the
new checker below and fixed there too.

### F23 `[CORRECTNESS]` F14's justification named a class with no such method

F14 cited `PostingsWriterBase.startDoc`'s `assert docID >= 0`.
`PostingsWriterBase` has no `startDoc` at all -- it is on
`PushPostingsWriterBase` -- and the real guard is not an assert. It is
`Lucene104PostingsWriter.startDoc`'s `if (docID < 0 || docDelta <= 0) throw new
CorruptIndexException("docs out of order (...)")` and `addPosition`'s
`if (position < 0) throw new CorruptIndexException("position=... is < 0")`.
Both are **production checks**, so F14 is a stronger `MISSING` finding than it
was written up as. Citations corrected.

### H1–H3 defensive hardening (no live defect, but the proofs rest on them)

Three bounds added that the existing code already survived — verified by
running the new unit tests against the unfixed code, where these two pass —
but which the batch's `// ARITH:` proofs name:

- **`check_arc_array_fits`**: a `bytesPerArc * numArcs` array physically
  precedes `posArcsStart`, so a header naming more than the bytes below it
  cannot describe a real node. Without it the reverse cursor reaches a negative
  position on every probe (caught by `read_byte`, but only after the search has
  been driven entirely by garbage). This is what bounds `numArcs` by the body
  length for every downstream proof.
- **`read_presence_bytes`** bounds the bit table the same way — direct
  addressing's counterpart, since its arc array holds only the *present* arcs
  and cannot be bounded by `numArcs`.
- **`BytesReader::skip_bytes`** now saturates rather than wrapping. Not a "make
  it fit" reflex: `i64::MIN`/`i64::MAX` are both outside `0..bytes.len()`, so
  the next `read_byte` reports corruption, whereas a wrap could land the cursor
  back *inside* the body and decode a plausible arc from the wrong offset.

### Byte-flip sweep

`crates/lucene-codecs/tests/fst_byte_flip_sweep.rs`. Unlike the `.fdm`/`.tvd`/
`.vemf` sweeps there is nothing to **re-sign**: an FST file carries a
`CodecUtil.writeHeader` and no footer at all (`FSTMetadata.save` writes no
checksum), so a flipped body byte already reaches the decoder on its own
merits. Header flips are swept too and are rejected by the header check.

Every byte, all eight bits, over the whole Lucene-written fixture corpus —
`fst`, `fst_deep_trie` (multi-node, multi-level, list-encoded),
`fst_binary_search`, `fst_direct_addressing`, `fst_continuous`, the three
`fst_seek_floor_backtrack_*`, `fst_seek_non_root_array_node`, `fst_byte2`,
`fst_byte4`, `fst_empty_key` — plus a 4 057-byte, 400-term FST from this
port's own builder, so the sweep spans many nodes and multi-byte `vlong`
targets rather than one 60-byte fixture. Each flip is driven through
`read`, `read_borrowed`, `get`/`get_labels`, `get_typed`,
`seek_ceil`/`seek_floor`/`seek_exact` in both key domains, a full ascending
enumeration in both, and the `IntsRef` enumeration.

| corpus | rejected / flipped |
|---|---|
| the 12 Lucene-written fixtures | 2 853 / 7 680 (37%) |
| a 4 KB, 400-term built FST | 4 183 / 32 456 (12.9%) |

The lower rate on the built FST is expected and is not a gap: most of that body
is label and output *payload*, where a flip names a different but perfectly
well-formed key or output. The bar the sweep enforces is the one that matters —
nothing panics, nothing aborts, nothing fails to terminate. It found F4, F7 and
F12 (and, before the enumeration cap was added, took a SIGKILL).

### Verdict

Swept clean. 98 lint sites resolved, 18 findings fixed, 3 hardening bounds
added, byte-flip sweep green over 40 136 flips.

---

## `postings_writer.rs`

Java: `codecs/lucene104/Lucene104PostingsWriter.java` +
`codecs/lucene103/blocktree/Lucene103BlockTreeTermsWriter.java`. 47 lint sites:
7 resolved as `wrapping_*` (Java `int` arithmetic the reader replays), 2 as new
validation, 38 with an `// ARITH:` proof.

This is a **writer**: its inputs are the caller's in-memory postings, not bytes
off disk, so the gate's risk profile is different — the failure mode is a panic
on a caller mistake rather than a corrupt-file abort. Two real gaps
nonetheless.

### F14 `[MISSING]` no bound on negative doc IDs or positions

`validate_field` checked sortedness and `freq >= 1` but never that the first
doc ID or the first position is non-negative — which
`PostingsWriterBase.startDoc`'s `assert docID >= 0` and
`FreqProxTermsWriterPerField` guarantee on Java's side. Without it a negative
doc ID reaches `docID - lastDocID` as an unbounded subtraction (and lands in
`.doc` as a delta no reader can undo), and a negative position does the same to
`position - lastPosition`. Both are now rejected by name
(`Error::NegativeDocId`, `Error::NegativePosition`), and they are exactly what
makes the delta proofs below hold: doc IDs in `0..=i32::MAX` strictly ascending
from `prev = -1`, positions in `0..=i32::MAX` strictly ascending from 0.

### F15 `[CORRECTNESS]` `PosSkipWriter`'s pointer deltas underflowed the fallback they were written for

`PositionLayout::sample` deliberately falls back to `0` for an out-of-range
block index rather than panicking, and its doc comment says the failure "should
be a wrong pointer a differential test catches, not a panic inside a writer".
But the caller then computed `s.pos_fp - self.level0_last_pos_fp` on `u64`s, so
a `0` from that fallback **underflows** — a panic in debug, three frames from
the code that was written to avoid one. Java's is a `long` subtraction that
wraps; now `wrapping_sub`, which is both the faithful port and the behaviour
the fallback assumes.

### The rest

The doc-delta accumulators (`doc_id - prev`, `lastDocID - prevDocID` at both
skip levels) are now `wrapping_sub`, matching Java's `int` subtraction bit for
bit; with F14's bound a valid caller never reaches the wrap. The proofs worth
naming:

- the two full-block/level-1 loops maintain `start <= docs.len()` because
  `start` only ever advances by the amount the loop condition just proved was
  left — so `len - start` cannot underflow and `start + N` cannot pass the end;
- the dense bit-set encoding's `s` accumulator is bounded by `doc_range - 1`
  and its word index by `ceil(doc_range / 64)`, and the branch is only reached
  when `doc_range < 32 * 256`, so the `-numBitSetLongs` token is in `-128..=-1`
  — exactly the range `read_full_block_header` decodes as a bit set;
- `totalTermFreq - docFreq` is non-negative because every per-doc freq is
  `>= 1` over exactly `docFreq` documents;
- the payload-byte cursors are bounded by the concatenation they index, which
  this same function built.

### Verdict

Swept clean. 47 lint sites resolved, 2 findings fixed.

---

## `vectors.rs`

Java: `Lucene99FlatVectorsReader/Writer`, `OffHeap{Float,Byte}VectorValues`,
`OrdToDocDISIReaderConfiguration`, `VectorUtil`. 29 lint sites: 6 resolved with
`checked_*`, 4 with `wrapping_*`, 19 with an `// ARITH:` proof.

### F16 `[CORRECTNESS]` `Math.multiplyExact` ported as a plain `*`

`Lucene99FlatVectorsReader.FieldEntry`'s compact constructor computes
`numBytes = Math.multiplyExact(Math.multiplyExact(dimension, byteSize), size)`
and lets the `ArithmeticException` escape. Ported as
`(dimension as i64) * (byte_size as i64) * (size as i64)`, which genuinely
overflows: `(2^31 - 1) * 4 * (2^31 - 1) > 2^63`. A debug build panics; a
release build wraps, and a wrapped `expected` is not merely a wrong number — it
is one an attacker picks to make the `expected == vectorDataLength` identity
hold for an absurd `dimension`/`size` pair. Now `checked_mul`, matching Java.

### F17 `[CORRECTNESS]` `len - FOOTER_LENGTH` on a file shorter than its footer

`open` computed both footer offsets as `buf.len() - codec_util::FOOTER_LENGTH`
with nothing establishing that either file is at least 16 bytes. Underflows to
a huge offset in release, panics in debug. Now a `checked_sub` reporting
corruption.

### F18 `[CORRECTNESS]` `raw_values` formed the very sum it was guarding

`let end = start + entry.vector_data_length as usize; if end > self.data.len()`
— the `a + b > len` shape `docs/arithmetic-gate.md` names, with `start` a
`vectorDataOffset` off disk. Rewritten as
`start.checked_add(len).and_then(|end| self.data.get(start..end))`.

### F19 `[CORRECTNESS]` the three byte-vector kernels panicked where Java wraps

`VectorUtil.squareDistance(byte[], byte[])` and friends accumulate into a Java
`int` and document the bound explicitly ("this will not overflow if dim <
2^18"). `dimension` comes off `.vemf` with no upper bound in either language —
Java cross-checks it against `FieldInfo`, which is itself unbounded on read —
so a `.vec` with a large `dimension` overflows the accumulator. Java wraps
silently; this port panicked in debug. The three accumulators (and
`dotProductScore`'s `a.length * (1 << 15)`) are now `wrapping_*`, which is
Java's semantics bit for bit; the per-element products are proved instead
(`|d| <= 255` so `d * d <= 65 025`).

No upper bound was *added* on `dimension`: `KnnVectorsFormat`'s 1 024 is an
indexing-time limit that a custom codec raises, so a cap here could reject a
file real Lucene wrote.

### The rest

`bytes`/`raw_range` take their slices with `get` rather than `[..]`; the
`ord * vector_bytes` product is proved against the
`slice.len() == size * vector_bytes` identity `read_field_entry` enforces (see
PERF-2 for why the proof, rather than a `checked_add`, is what belongs on that
path). `validate_field` and `merge_one_flat_vector_field` size their products
with `checked_mul`.

### Byte-flip sweep

`every_resigned_single_byte_vemf_and_vec_corruption_is_an_error_or_a_clean_decode`,
over a three-field, 600-document segment chosen to span every structure the
pair has: a **dense** float field (no ord→doc structures), a **sparse** float
field (an `IndexedDISI` bitset *and* a `DirectMonotonic` ordinal→doc sequence,
both in `.vec` after the vectors) and a **byte** field (a different `byteSize`
and therefore a different `alignOutput` padding). Every flip is driven through
`open`, every vector of every field, `ordToDoc` for every ordinal, the bulk
`raw_range`, an exhaustive KNN search and a full doc→ordinal cursor walk.

**465 / 26 168** rejected overall; **350 / 508 (69%)** for the `.vemf` alone.
The low overall figure is the expected one: nearly every `.vec` byte is a
vector *component*, and flipping one yields a different but perfectly
well-formed float.

### Verdict

Swept clean. 29 lint sites resolved, 4 findings fixed.

---

## `hnsw.rs` and `hnsw_vectors.rs`

### INTENTIONAL-1 two modules were already clean

Both were already lint-clean, and the hand-audit of the half the lint does not
cover found **nothing new**. That is a real result, recorded rather than padded
into findings — c24 found four such modules.

What was actually wrong was the *paperwork*: seven `#[allow]`s in `hnsw.rs`
sat immediately after a `}` rather than after their comment block, so
`scripts/check-arith-allows.py` could not see the `ARITH:` proof that covered
the site above them. Each now carries its own one-line proof; each was checked
against its boundary value first (the two `len() as i64 - 1` sites bottom out
at -1 rather than underflowing; `cursor -= 1`/`i -= 1` are guarded by the loop
conditions above them; `new_gain` is bounded by `k + degree <= i32::MAX / 2`).

The non-lint half was checked in full and is genuinely sound:
`hnsw_vectors::read_field_entry` grows `nodes_by_level` one level at a time
rather than pre-allocating `numLevels` (Java's `new int[numLevels][]` is a
51 GB reservation here), bounds each level's node count by the `.vem` bytes
left, and bounds the graph and node-offset regions in `u128`;
`neighbors_into` caps `arcCount` at `2 * maxConn` before touching the scratch
buffer and range-checks every decoded ordinal against `size`; `compute_join_set`
calls `check_neighbors` after both graph walks; `merge_graphs` bounds both the
ordinal maps and the `initialized_nodes` bitset against the graphs they will
index. The one remaining amplification — `OnHeapHnswGraph::with_size`'s
`vec![Vec::new(); numNodes]` at 24 bytes a node — is sized from
`merged_ord_to_doc.len()`, i.e. memory the caller has already committed, not
from a number read off a file; Java's `new NeighborArray[numNodes][]` is the
same shape at 8 bytes a node.

The `.vem`/`.vex` byte-flip sweep already exists
(`every_resigned_single_byte_vem_and_vex_corruption_is_an_error_or_a_clean_decode`,
**1 656 / 3 536** over a 300-vector multi-level graph) and stays green.

### Verdict

Both swept clean, no runtime change. 7 proofs written.

---

## Performance

Measured with c24's min-of-40 harness (criterion's mean is unusable on this
machine), alternating A/B three times inside the container, where **B** is the
same tree with every check this batch added stripped out. The workloads are the
three paths the changes actually touch: the `.tip` FST lookup and enumeration,
and the per-candidate vector fetch that HNSW and brute-force KNN both score
through.

| workload | A (checked) | B (unchecked) | delta |
|---|---|---|---|
| `Fst::get` × 2 000 keys | 627–629 µs | 626–646 µs | neutral |
| `FstEnum` full enumeration, 2 000 keys | 207–209 µs | 214–220 µs | neutral |
| `FstEnum::seek_ceil` × 2 000 | 2 769–2 862 µs | 2 813–3 002 µs | neutral |
| `exhaustive_search` k=10 over 20 000 × 32-dim | 143.5–144.1 µs | 148.5–149.6 µs | neutral |
| `vector_into` scan, 20 000 × 32-dim | 53.8–55.4 µs | 51.6–56.6 µs | neutral |

Every check added on a per-value path is hoisted out of its loop, as the brief
required: the node-header validations run once per *node* visited, not per arc
or per byte; `check_descent` is one comparison per level descended;
`checked_output_len` is one per output read; `RawVectorValues` computes
`dimension * byteSize` once when the field is opened rather than at every
`vector(ord)`.

Two regressions were found and removed rather than accepted:

### PERF-1 the first cut of F7 cost 7% of a full FST enumeration

Validating the label range *per emitted key* inside `labels_to_bytes` measured
225 µs against 210 µs unchecked; expressed as a fallible
`map(..).collect::<Result<_>>()` it was worse still (258 µs), because the
fallible collect cannot use the slice's size hint. Moving the bound to the two
places the label is *formed* (F7's `check_label_range` and
`bit_table_next_bit_set`) made the narrowing provably total and therefore free
— and is the stronger rejection, since it refuses the node rather than the key.

### PERF-2 `ok_or` on a wide error enum cost 40% of the vector fetch

`self.slice.get(start..end).ok_or(Error::OrdOutOfRange(ord, size))` builds the
error value **on the success path**, and `vectors::Error` carries `String`
variants, so it is wide: 78 µs against 55 µs on the 20 000-vector scan. Spelled
as a `match` (`ok_or_else` reads better but trips
`clippy::unnecessary_lazy_evaluations`, which does not know the enum's size).
A `checked_add` for `start + vector_bytes` on the same line cost a further 6%
and was replaced by the proof that makes it unnecessary — the
`slice.len() == size * vector_bytes` identity `read_field_entry` enforces.

## Verifying no new bound rejects a file real Lucene wrote

Every bound added this batch is traced to the Java writer rather than invented:

| bound | why a real writer cannot violate it |
|---|---|
| `numArcs >= 1`, `bytesPerArc >= 1` | `FSTCompiler.writeNodeForBinarySearch` needs `FIXED_LENGTH_ARC_SHALLOW_NUM_ARCS` arcs to be chosen at all, and every arc slot is at least one byte |
| `bytesPerArc * numArcs <= posArcsStart + 1` | the array is physically written below `posArcsStart` |
| `getNumPresenceBytes(numArcs) <= bitTableStart + 1` | likewise for the bit table |
| `firstLabel + numArcs <=` the `INPUT_TYPE` alphabet | there are no labels outside it to put in the range |
| `nextBitSet(..) < numArcs` | `FSTCompiler` zeroes the bit table's padding bits |
| arc targets strictly precede their own node | `FSTCompiler` writes nodes bottom-up into a forward-growing `BytesStore` |
| label `>= 0` | labels come from an `IntsRef` of byte values or code points; `END_LABEL == -1` is synthetic |
| vint ≤ 5 bytes / vlong ≤ 10 | `DataOutput.writeVInt` emits "between one and five bytes", `writeVLong` "between one and nine" and rejects negatives — the reader's bound matches the first exactly and is one byte looser than the second; 10.5.0's *reader* has no bound at all (F22) |
| output length ≤ `pos + 1` | the reverse cursor consumes the output downward from `pos` |
| doc IDs and positions `>= 0` | `PostingsWriterBase.startDoc`'s assert; `FreqProxTermsWriterPerField` |

Empirically: all 12 Lucene-written FST fixtures (every node encoding, both wide
label types, the empty-key metadata branch), every `blocktree` fixture (which
reads a real `.tip` FST), and **`scripts/verify-write-path.sh` 22/22** — in
which unmodified Lucene 10.5.0 reads back the `.fst`, `.vec`/`.vemf` and
`.vem`/`.vex` bytes this port writes — all pass unchanged.

## What the Tier-2 review changed

Four of this batch's own findings (F20–F23) came from the review, and three of
them are the same failure mode `docs/arithmetic-gate.md` warns about: **a
proof whose stated reasoning is wrong even where the conclusion survives**.
This is the fourth consecutive batch to hit it (c19, c24, c27, c31), which is
why two of the fixes are mechanical rather than editorial:

- **A new standing rule**, beside the `FixedBitSet` one: *check an
  out-of-domain sentinel at every call site, and record the check per call
  site, not per function.* F20 is exactly that shape — the sweep report said
  "the sentinel is handled" when one of two call sites handled it — and the
  section spells out why a byte-flip sweep structurally cannot find it.
- **`scripts/check-java-refs.py`**, new, wired into `scripts/gate.sh`,
  `.githooks/pre-commit` (which defers to it) and CI. It resolves every
  ``ClassName.methodName`` and quoted Java message in a comment against the
  **pinned** tree, and skips cleanly when that tree is absent. It found F22's
  twin in `terms_dict.rs` and F23 on its first run. Scope is incremental like
  the arithmetic gate's: it enforces the audited file list in `AUDITED` (242
  citations, clean) and `--all` reports the workspace backlog (84 citations,
  mostly paraphrases naming a method that lives on a parent or sibling class)
  as a follow-up.

The review also confirmed as sound: `check_descent`, `read_array_node_header`,
`check_label_range`, F5's `>>>` ports, F16's `multiplyExact` ordering, the
dense bit-set proof and all seven `hnsw` proofs.

## An incidental find: `lucene-index`'s gate was red the same way

Fixing `check-arith-allows.py` to cross-check *every* crate's burn-down row —
it previously skipped any crate whose marker count was zero, which is blind in
exactly the direction that matters — immediately reported `lucene-index`:
table claiming 3 pending modules, tree carrying none. `cargo clippy -p
lucene-index --all-targets` confirms **68 live `arithmetic_side_effects`
errors** in precisely the three modules the table names (`index_writer` 22,
`merge` 14, `merge_policy` 32). Its markers had been deleted without the audit,
the same state this batch found `lucene-codecs` in.

Not this batch's crate to audit, so the three `// TODO(arith-audit)` markers
are restored on their `mod` declarations — the documented opt-out — which
returns that crate's gate to green *honestly* rather than by pretending the
modules are audited. `cargo clippy -p lucene-index --all-targets -- -D
warnings` now passes. The audit itself remains open.

## Gates

- `cargo clippy -p lucene-codecs --all-targets -- -D warnings`: **0 errors**
  (was 197).
- `cargo test -p lucene-codecs`: green. `-p lucene-index -p lucene-search
  -p lucene-core` (the crates downstream of the `Outputs::decode` and
  `decode_ints_key` signature changes): green.
- `scripts/verify-write-path.sh`: **22/22**.
- `python3 scripts/check-parity.py`: ok. `python3 scripts/check-arith-allows.py`:
  ok (3 modules still unaudited, all in `lucene-index`).
- `cargo llvm-cov -p lucene-codecs --summary-only`: no file below 95% lines.
  Touched files: `fst.rs` 97.47%, `hnsw.rs` 95.94%, `hnsw_vectors.rs` 95.22%,
  `postings_writer.rs` 99.39%, `vectors.rs` 96.79%, `suggest.rs` 98.95%.

One pre-existing warning outside this batch's scope is left alone:
`lucene-index/src/check_index.rs:11824` (`&mut Vec` where `&mut [_]` would do,
in a test helper belonging to in-flight work on that crate).
