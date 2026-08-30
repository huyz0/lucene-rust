# b4-fst-blocktree

Files swept: `crates/lucene-codecs/src/{fst.rs, blocktree.rs, terms_dict.rs}`.

Lucene source read: `/home/tuong/work/lucene` at 10.5.0.

**Version note, checked with `grep` rather than assumed.** `lucene/core` at
10.5.0 has **no** `org/apache/lucene/codecs/lucene90/blocktree` package: it
was replaced by `org/apache/lucene/codecs/lucene103/blocktree`, and the
`lucene90` classes now live in `lucene/backward-codecs`. The batch prompt
named `lucene90`; `docs/parity.md` names `lucene103`, and `lucene103` is what
this port actually targets and what was compared against. The difference is
not cosmetic: `lucene90`'s `.tip` term index is an FST, `lucene103`'s is a
purpose-built binary trie (`TrieReader`/`TrieBuilder`). That is decisive for
the "is `fst.rs` dead code" question below.

---

## `crates/lucene-codecs/src/fst.rs`

**Java counterparts:**
`org/apache/lucene/util/fst/FST.java`, `FSTEnum.java`, `BytesRefFSTEnum.java`,
`IntsRefFSTEnum.java`, `BitTableUtil.java`, `Outputs.java`,
`ByteSequenceOutputs.java`, `PositiveIntOutputs.java`, `PairOutputs.java`,
`Util.java`, `ReverseBytesReader.java`, `OnHeapFSTStore.java`,
`OffHeapFSTStore.java`, `FSTCompiler.java`, `FSTSuffixNodeCache.java`.

### Method correspondence

| Rust | Java | Status |
|---|---|---|
| `BytesReader::{new,get_position,set_position,skip_bytes,read_byte,read_bytes}` | `ReverseBytesReader` | identical (position semantics, negative `skipBytes`) |
| `BytesReader::{read_vint,read_vlong}` | `DataInput.readVInt/readVLong` | identical modulo the too-many-bits throw (wraps instead; unreachable before EOF) |
| `BytesReader::{read_output,skip_output}` | `ByteSequenceOutputs.read`/`skipOutput` | identical |
| `num_presence_bytes` | `FST.getNumPresenceBytes` | identical |
| `bit_table_is_bit_set` / `_count_bits` / `_count_bits_up_to` / `_next_bit_set` / `_previous_bit_set` | `BitTableUtil.isBitSet`/`countBits`/`countBitsUpTo`/`nextBitSet`/`previousBitSet` | identical results; byte-at-a-time instead of Java's 8-byte `readLong` fast path (INTENTIONAL, F7) |
| `output_add` | `ByteSequenceOutputs.add` | identical (`is_empty()` stands in for Java's `== NO_OUTPUT` identity check; `read` returns the singleton for len 0) |
| `Arc` + accessors | `FST.Arc<T>` | identical fields for the four node encodings |
| `target_has_arcs` | `FST.targetHasArcs` | identical |
| `read_fst_metadata_prefix` | `FST.readMetadata` | **was divergent — F1, fixed** |
| `Fst::read` | `FST(FSTMetadata, DataInput)` + `OnHeapFSTStore` | identical |
| `Fst::read_borrowed` | `OffHeapFSTStore` cost model | not-in-Java as a method; deliberate zero-copy sibling |
| `Fst::read_label` | `FST.readLabel` | identical (all three `INPUT_TYPE`s, version-gated `BYTE2` byte swap) |
| `Fst::first_arc` | `FST.getFirstArc` | identical |
| `Fst::read_arc` | `FST.readArc` | identical |
| `Fst::seek_to_next_node` | `FST.seekToNextNode` | identical |
| `Fst::read_presence_bytes` | `FST.readPresenceBytes` | identical |
| `Fst::read_arc_by_direct_addressing` | `FST.readArcByDirectAddressing(Arc,BytesReader,int,int)` | identical |
| `Fst::read_arc_by_index` | `FST.readArcByIndex` | identical |
| `Fst::read_arc_by_continuous` | `FST.readArcByContinuous` | identical |
| `Fst::read_last_arc_by_direct_addressing` / `_by_continuous` | `FST.readLastArcByDirectAddressing`/`readLastArcByContinuous` | identical |
| `Fst::read_last_target_arc` | `FST.readLastTargetArc` | identical (incl. list-node backscan and the `skipBytes(-1)` undo) |
| `Fst::read_next_arc_label` | `FST.readNextArcLabel` | identical |
| `Fst::read_next_real_arc` | `FST.readNextRealArc` | identical |
| `Fst::read_first_real_target_arc` | `FST.readFirstArcInfo` + `readNextRealArc` | identical |
| `Fst::read_first_target_arc` | `FST.readFirstTargetArc` | identical |
| `Fst::read_next_arc` | `FST.readNextArc` | identical (error instead of `IllegalArgumentException`) |
| `Fst::find_target_arc` (list / DA / continuous branches) | `FST.findTargetArc` | identical |
| `Fst::find_target_arc_binary_search` | `FST.findTargetArc`'s `ARCS_FOR_BINARY_SEARCH` branch | **was divergent — F2, fixed** |
| `Fst::find_arc_binary_search` | `Util.binarySearch` | identical (`low = arc.arcIdx`) |
| `Fst::find_next_floor_arc_{binary_search,direct_addressing,continuous}` | `FSTEnum.findNextFloorArc*` | identical |
| `FstEnum::{push_first,push_last,rewind_prefix,rollback_to_last_fork_then_push,backtrack_to_floor_arc}` | `FSTEnum.pushFirst`/`pushLast`/`rewindPrefix`/`rollbackToLastForkThenPush`/`backtrackToFloorArc` | identical |
| `FstEnum::seek_ceil_{list,binary_search,direct_addressing,continuous}` + `do_seek_ceil` | `FSTEnum.doSeekCeilList`/`doSeekCeilArrayPacked`/`ArrayDirectAddressing`/`ArrayContinuous`/`doSeekCeil` | identical |
| `FstEnum::seek_floor_*` + `do_seek_floor` | `FSTEnum.doSeekFloor*` | identical |
| `FstEnum::do_seek_exact` | `FSTEnum.doSeekExact` | identical |
| `FstEnum::advance` | `FSTEnum.doNext` | identical walk; fused-after-failed-seek (F5, INTENTIONAL) |
| `FstEnum::{seek_ceil,seek_floor,seek_exact,next}` | `BytesRefFSTEnum` | identical |
| `FstEnum::*_labels`, `IntsRefFstEnum` | `IntsRefFSTEnum` | identical |
| `Fst::{get,get_labels,seek_exact,seek_exact_labels}` | `Util.get(FST,BytesRef)` / `Util.get(FST,IntsRef)` | identical |
| `build_fst` / `build_node` / `NodeHash` / `write_fst` | `FSTCompiler` / `FSTSuffixNodeCache` / `FSTMetadata.save` | deliberately simplified builder (see parity.md); `write_fst`'s metadata half **was divergent — F1, fixed** |
| `Outputs` / `PositiveIntOutputs` / `ByteSequenceOutputs` / `PairOutputs` | `Outputs<T>` and friends | Rust-only layering, **not** wire-compatible (F6, INTENTIONAL, now documented) |
| — | `Util.shortestPaths` / `TopNSearcher` / `readCeilArc` / `toDot` / `dotToFile` | **not ported** (F8) |
| — | `NoOutputs`, `CharSequenceOutputs`, `IntSequenceOutputs` | not ported, no caller |

### Findings

#### F1 [CORRECTNESS] `FSTMetadata.emptyOutput` was missing its `writeFinalOutput` framing — fixed

**Java.** `FSTMetadata.save` serializes the empty-string key's output through
`outputs.writeFinalOutput` (for `ByteSequenceOutputs`: `vint(len)` then the
payload), reverses **that whole serialized buffer**, and writes
`vint(reversedLen)` followed by it. `FST.readMetadata` mirrors that: it copies
`numBytes` into a store, positions a reverse `BytesReader` at `numBytes - 1`,
and calls `outputs.readFinalOutput`, which consumes the length prefix.

**This port.** `read_fst_metadata_prefix` read `numBytes`, reversed the buffer
and kept it verbatim; `write_fst` reversed the raw output and length-prefixed
that.

**Consequence.** Both directions broken against real Lucene, and invisible to
self-round-trip because the two halves made the same mistake. Reading a real
Lucene FST that accepts the empty string yielded `vint(len) ++ payload` as the
empty output — verified empirically: the 10.5.0-written fixture's metadata is
`01 0c 54 55 50 54 55 4f 2d 54 4f 4f 52 0b`, whose reverse is
`0b "ROOT-OUTPUT"`, and this port returned the leading `0b`. Writing, real
Lucene read the payload's last byte as a length and produced garbage.

**Why no existing test caught it.** No `GenFst*` generator built an FST that
*accepts* the empty string (`GenFst.java` lists `""` among its **absent**
keys), and the one hand-built unit test encoded the wrong layout in the test
itself.

**Fixed:** `fst.rs::read_fst_metadata_prefix` (decode through a reverse
`BytesReader` + `read_output`, exactly as Java does) and `fst.rs::write_fst`
(serialize, then reverse, then length-prefix).
**Tests:** new Java-written fixture `fixtures/src/GenFstEmptyKey.java` →
`fixtures/data/fst_empty_key/`, read by
`crates/lucene-codecs/tests/fst_empty_key_fixtures.rs` (5 tests: metadata
value, every present key incl. `""`, absent keys, full enumeration, and
`write_fst(read(bytes)) == bytes` against the real Lucene bytes). Reverse
direction: `write_fst_fixture` gained an `empty/` fixture and `VerifyFst.java`
verifies it through real `FST.read`/`Util.get` — ran green
(`VerifyFst OK (empty): 5 present keys resolved, 6 absent keys rejected`).
The stale unit test `read_accepts_empty_string_via_empty_output` was corrected
to the real layout. Reverting the fix fails 4 of the 5 new tests.

#### F2 [CORRECTNESS] `find_target_arc`'s binary-search branch left `Arc.arcIdx` unset — fixed

**Java.** `FST.findTargetArc`, on a hit at slot `mid`, does
`arc.arcIdx = mid - 1; return readNextRealArc(arc, in);` — so `arcIdx` ends at
`mid`.

**This port.** It seeked to the slot and decoded it directly, leaving
`arc_idx` at `Arc::default()`'s `0`.

**Consequence.** `arcIdx` is the cursor `readNextRealArc`/`readNextArcLabel`
use to find the *following* slot. `FstEnum::seek_exact` stores the arc
`find_target_arc` returned, so a subsequent `next()` resumed from slot 1 of
that node no matter which key was sought. Measured on the real
`fst_binary_search` fixture: `seek_exact([0x28])` then `next()` re-yielded
`[0x28]` before continuing — a duplicated key. Only reachable through a real
`FSTCompiler`-built fixed-length-arc node; this port's own `build_fst` emits
only list-encoded nodes, which take a different `read_next_real_arc` branch,
so no self-built test could ever have found it.

**Fixed:** `arc_idx: mid - 1` + `read_next_real_arc`, i.e. Java's own code
path rather than a duplicate of its address arithmetic.
**Test:** `fst_binary_search_fixtures.rs::seek_exact_then_next_resumes_from_the_sought_key_through_a_binary_search_node`,
which seeks every key of the fixture in turn and asserts the remainder of the
enumeration. Reverting the fix fails it.

#### F3 [INTENTIONAL] `node_flags` is `0` for list nodes, not the first arc's flags byte

Java's `readFirstArcInfo` leaves `arc.nodeFlags` holding the first arc's flags
byte for list-encoded nodes; this port zeroes it. Equivalent, because arc
flags only use bits 0..5 and `BIT_ARC_HAS_FINAL_OUTPUT` is only ever set
together with `BIT_FINAL_ARC` — so no arc flags byte can equal
`ARCS_FOR_BINARY_SEARCH` (0x20), `ARCS_FOR_DIRECT_ADDRESSING` (0x40) or
`ARCS_FOR_CONTINUOUS` (0x60), which is exactly the invariant that makes the
sentinel encoding work at all. `read_last_target_arc` keeps Java's assignment.
Verified, not assumed. No change.

#### F4 [PERF, in this port's favour] `read_borrowed`

Not a divergence to fix; recorded so the batch's PERF column isn't silently
empty. `Fst::read` matches `OnHeapFSTStore` (one copy of the body);
`Fst::read_borrowed` has no Java counterpart as an API and gives
`OffHeapFSTStore`'s cost model over this port's mmap `Directory` with zero
allocation for the body. Already covered by
`read_borrowed_body_is_a_slice_not_a_second_owned_buffer` and
`read_borrowed_over_a_real_mmap_directory_input`.

#### F5 [INTENTIONAL] a failed seek leaves the enum fused

`BytesRefFSTEnum.seekCeil` returning `null` leaves `upto == 0`, which
`doNext` reads as "not started", so Java's next `next()` **restarts from the
first key**. This port sets `done = true` and stays exhausted. Restarting
after yielding `None` is the wrong contract for a Rust `Iterator`, and no
caller wants it. Recorded and now documented on `seek_ceil_labels`; no change.

#### F6 [INTENTIONAL] the typed `Outputs` layer is not wire-compatible with `Outputs<T>`

Lucene writes each arc's output through the FST's own `Outputs<T>` — a bare
`vlong` for `PositiveIntOutputs`, `A` then `B` for `PairOutputs<A, B>`. This
port's reader is hardcoded to `ByteSequenceOutputs`' `vint`-length framing and
the typed codecs encode *inside* that payload, and `PairOutputs::encode` adds
its own `vint` first-component length that Java does not write. So a
`build_fst_typed::<PositiveIntOutputs>` FST is a valid `FST<BytesRef>`, not a
valid `FST<Long>`. Harmless today — `suggest.rs` is the only consumer and both
writes and reads its own FSTs — but it was undocumented, which is the part
that was wrong. Now stated at the `Outputs` trait definition. Making the
reader generic over `Outputs` is the prerequisite for interop and is
deliberately out of scope.

#### F7 [PERF] `BitTableUtil` reads the presence table a byte at a time

Java reads 8 bytes at a time (`readLong` + `Long.bitCount`); this port loops
bytes. Worst case is `countBits` over a 256-label direct-addressing node: 32
byte reads vs 4 word reads. Not fixed, and this is a judgement rather than an
oversight: the bit-table is at most 32 bytes, it is read from an already
in-memory slice, and `BytesReader` is a *reverse* cursor, so a word-at-a-time
version cannot simply `u64::from_le_bytes` a slice — it would need its own
direction-aware path, i.e. a second implementation of the primitive to keep in
agreement forever. The batch's measured hot spot is elsewhere (finding A1);
revisit if an FST profile ever says otherwise.

#### F8 [MISSING, out of scope] `Util.shortestPaths`/`TopNSearcher`, `readCeilArc`, `toDot`

Not ported. `toDot`/`dotToFile` are debug output. `shortestPaths`/
`TopNSearcher`/`readCeilArc` are the N-best-completions search real Lucene's
suggesters use; `suggest.rs` has its own `top_n_completions` instead. Recorded
rather than ported: it belongs with the suggester work in batch b8, and
porting it here would have no caller to validate it against.

#### D2 revisited — `fst.rs` is *not* dead code awaiting the term dictionary

The prior sweep (`docs/sweep/findings.md`, D2) recorded that `fst.rs` is
complete, fixture-verified, and unused by the read path, and framed that as a
consequence of `blocktree.rs`'s eager materialization — "real block-tree
navigation would route through it".

Re-checked, and **that framing is wrong**. In Lucene 10.5 the `.tip` term
index is a `TrieReader`, not an FST, and
`grep -rl "util\.fst" lucene/core/src/java` returns nothing outside
`util/fst/` itself — no codec in `lucene/core` references FST at all. Its
users are `lucene/suggest`, `lucene/codecs`, `backward-codecs` and `sandbox`.
So a fully lazy block-tree navigator in this port would *still* never call
`fst.rs`, and `fst.rs`'s only consumer being `suggest.rs` mirrors Lucene
exactly rather than indicating a gap. The reachability claim in D2 stands; its
explanation does not. Corrected in `blocktree.rs`'s module doc.

### Verdict

Swept clean apart from the recorded items. Two correctness defects found and
fixed, both proven by real-Lucene fixtures in both directions; F5/F6/F7 are
recorded divergences with reasons; F8 is deferred to b8 with a named owner.

---

## `crates/lucene-codecs/src/blocktree.rs`

**Java counterparts:** `org/apache/lucene/codecs/lucene103/blocktree/`
`TrieReader.java`, `TrieBuilder.java` (`ChildSaveStrategy` only),
`Lucene103BlockTreeTermsReader.java`, `FieldReader.java`,
`SegmentTermsEnum.java`, `SegmentTermsEnumFrame.java`,
`CompressionAlgorithm.java`, `IntersectTermsEnum.java`,
`IntersectTermsEnumFrame.java`, `Stats.java`, plus
`org/apache/lucene/util/compress/LowercaseAsciiCompression.java`.

### Method correspondence

| Rust | Java | Status |
|---|---|---|
| `read_u64_at` / `read_u8_at` / `read_u64_n_bytes` | `RandomAccessInput.readLong`/`readByte`, `TrieBuilder.writeLongNBytes`'s inverse | identical |
| `load_node` (`SIGN_NO_CHILDREN` arm) | `TrieReader.loadLeafNode` | identical |
| `load_node` (single-child arms) | `TrieReader.loadSingleChildNode` | identical |
| `load_node` (`SIGN_MULTI_CHILDREN` arm) | `TrieReader.loadMultiChildrenNode` | identical |
| `multi_children_labels_and_fps` | `TrieReader.lookupChild` + `ChildSaveStrategy.{BITS,ARRAY,REVERSE_ARRAY}.lookup` | same decodings, generalized from "find one label" to "enumerate all" (INTENTIONAL, entailed by A1) |
| `expand_floor` | `SegmentTermsEnumFrame.setFloorData` + `scanToFloorFrame` | same byte layout; decodes every floor block instead of selecting one (INTENTIONAL, entailed by A1) |
| `collect_leaf_blocks` | `SegmentTermsEnum`'s trie descent | eager full traversal (A1) |
| `decode_block_at_depth` (header) | `SegmentTermsEnumFrame.loadBlock` | identical; **B2 fixed** (allocation guards) |
| `decode_block_at_depth` (entry loop) | `SegmentTermsEnumFrame.nextLeaf`/`nextNonLeaf` | identical |
| `decode_block_at_depth` (stats/meta) | `SegmentTermsEnumFrame.decodeMetaData` | identical (singleton run length, DOCS aliasing, per-block `absolute`) |
| `decompress_lowercase_ascii` | `LowercaseAsciiCompression.decompress` | identical |
| compression dispatch | `CompressionAlgorithm.byCode` + `read` | identical, incl. rejecting code 3 |
| `read_bytes_ref` | `Lucene103BlockTreeTermsReader.readBytesRef` | identical + negative-length guard |
| `read_freq_pair` | reader ctor's `sumTotalTermFreq`/`sumDocFreq` DOCS aliasing | identical |
| `open` (headers, per-field metadata, validation) | `Lucene103BlockTreeTermsReader` ctor + `FieldReader` ctor | identical; **B1 fixed** (length equality) |
| `FieldTerms::seek_exact` | `SegmentTermsEnum.seekExact` | same answer, different algorithm (A1) |
| `TermsEnum::{next,seek_ceil,current}` | `SegmentTermsEnum.next`/`seekCeil`/`term` | same answers over a materialized `Vec` (A1) |
| `FieldTerms::{intersect,fuzzy_intersect,regexp_intersect}` | `FieldReader.intersect` → `IntersectTermsEnum` | **B3**: prefix-narrowed linear scan, no automaton |
| `FieldTerms::{postings,lazy_postings,positions,positions_for_docs,positions_flat}` | `SegmentTermsEnum.postings`/`impacts` | swept in b5 (`postings.rs`); glue here |
| `BlockTreeFields::{field,iter_fields,empty}` | `Lucene103BlockTreeTermsReader.terms`/`iterator` | identical intent |
| `prefix_upper_bound` | — | not-in-Java (helper for the prefix range) |
| — | `SegmentTermsEnum.seekExact(BytesRef, TermState)` / `termState()` / `ord()` | **not ported** (B4) |
| — | `SegmentTermsEnumFrame.prefetchBlock`, `SegmentTermsEnum.prefetch` | **not ported** (B5) |
| — | `Stats.java`, `FieldReader.getStats` | **not ported** (B6) |
| — | `IntersectTermsEnum` / `IntersectTermsEnumFrame` | **not ported** (B3) |

### Findings

#### B1 [MISSING] the recorded `.tip`/`.tim` lengths were only half-checked — fixed

**Java.** `CodecUtil.retrieveChecksum(IndexInput, long expectedLength)`
rejects **both** `in.length() < expectedLength` ("truncated file") and
`in.length() > expectedLength` ("file too long"), then verifies the footer.

**This port.** `open` checked only `index_length > tip.len()`, so a `.tmd`
recording a length *shorter* than the file it describes was accepted, and the
footer was then read from the wrong end of the file.

**Fixed:** by routing both files through `codec_util::retrieve_checksum_with_expected_length` (b1's port of `CodecUtil.retrieveChecksum(IndexInput, long)`) instead of the plain overload plus a hand-rolled one-sided comparison -- which also discharges part of the ledger's "lucene-codecs privately duplicates lucene-store primitives" carry-over for this call site.
**Test:** `recorded_tip_or_tim_length_shorter_than_the_file_is_rejected`
(appends one byte to each of `.tip`/`.tim` in turn — Lucene's "file too long"
case) — and, notably, four in-module test builders had to be corrected in the
same change: they wrote `indexLength` **excluding** the `.tip` footer, the
exact defect the write path had already been fixed for
(`docs/sweep/findings.md` O29). Only the one-sided check had kept them
passing. Every real-Lucene blocktree fixture (`blocktree_fixtures`,
`_multilevel`, `_compressed`, `_deep_nesting` — 30 tests) passes unchanged
under the strict check, which is the evidence that the writer, not the
reader, was already right.

#### B2 [CORRECTNESS] corrupt block lengths aborted the process instead of erroring — fixed

**Java.** `loadBlock` allocates `new byte[numBytes]`, which throws
(`NegativeArraySizeException` / `OutOfMemoryError`) on a corrupt length; a
caller can catch it.

**This port.** `num_stat_bytes`/`num_meta_bytes` were `read_vint()? as usize`
— a negative `vint` becomes ~2^64 — and all four regions went into
`vec![0u8; n]`, which **aborts** the process on allocation failure. An abort
cannot be caught at the FFI boundary, which is precisely the hazard
`data_input.rs` already documents for `readString`.

**Fixed:** `read_region_len` (sign + `remaining()` check) for the stats and
metadata regions and for the non-`allEqual` suffix-lengths region; a
`remaining()` check for the *uncompressed* suffix region; and `try_zeroed`
(fallible `try_reserve_exact`) for the suffix regions, whose length is the
*decompressed* size and so legitimately can exceed the file.
**Tests:** `decode_block_rejects_region_lengths_larger_than_the_file`
(`i32::MAX` and `-1`) and
`decode_block_rejects_uncompressed_suffix_length_past_the_file`.

#### B3 [MISSING → recorded, with a measurement and a blocking dependency] `intersect()` is a prefix scan, not an automaton intersection

**Java.** `FieldReader.intersect(CompiledAutomaton, BytesRef)` returns an
`IntersectTermsEnum` that walks the `.tip` trie and `.tim` blocks driven by a
`ByteRunnable` + `TransitionAccessor`, pushing a frame per trie level, killing
whole sub-blocks whose label leads to a dead automaton state, and rejecting
terms early with `commonSuffixRef`.

**This port.** `intersect`/`fuzzy_intersect`/`regexp_intersect` binary-search
the materialized, sorted `entries` down to the pattern's literal-prefix range
and then test every term in that range.

**Measured over-scan.** Real `Lucene103BlockTreeTermsWriter` fixture
`blocktree_multilevel_index`, field `many`, 8,000 distinct terms:

| pattern | literal prefix | terms scanned | terms matched | over-scan |
|---|---|---|---|---|
| `ab*` | `ab` | 11 | 11 | 1.0x |
| `a*` | `a` | 303 | 303 | 1.0x |
| `abc*` | `abc` | 0 | 0 | — |
| `a*z` | `a` | 303 | 9 | 34x |
| `a?c*` | `a` | 303 | 8 | 38x |
| `ab?d*` | `ab` | 11 | 0 | — |
| `*ing` | (none) | 8,000 | 1 | 8,000x |
| `?????` | (none) | 8,000 | 895 | 8.9x |

**Reading the numbers honestly.** Purely-prefix patterns already scan exactly
the range Lucene's automaton would reach — 1.0x, no gap. For patterns whose
automaton constrains *interior* positions the port over-scans by tens; for
`.*`-leading patterns it scans the field, but so does Lucene (a leading `.*`
has no dead state at any prefix, so `IntersectTermsEnum` visits every term
too, rejecting them with a per-term `commonSuffix` byte compare). The
intersect-time gap is therefore real but bounded — tens, not orders of
magnitude — on this corpus.

**Where the real gap is: not here.** Lucene never *loads* the blocks its
automaton rules out. This port has already decoded, allocated and sorted every
block of the field before any pattern is seen (finding A1). Automaton-driven
block skipping cannot pay for itself until that changes, which is the honest
reason not to port it now.

**Why it is not tractable in this batch.** `IntersectTermsEnum` (572 lines) +
`IntersectTermsEnumFrame` (324) need two things this port does not have:
1. `org.apache.lucene.util.automaton` — `Automaton`, `CompiledAutomaton`,
   `ByteRunAutomaton`, `Transition`, determinization, `commonSuffixRef`. None
   of it exists here: `regexp.rs` is a recursive backtracking matcher and
   `wildcard.rs` a token matcher; neither can enumerate transitions or report
   a dead state. That is batch **b8**'s subject matter.
2. Lazy `SegmentTermsEnumFrame`-style navigation — finding **A1**, a milestone
   of its own.

Recorded, with the dependency chain named, rather than half-built. Nothing was
changed.

#### A1 revisited [PERF] — the eager whole-field materialization still holds

Re-checked, not assumed: `open()` still calls `collect_leaf_blocks` over the
whole trie, expands every floor block, decodes every `.tim` block and sorts a
`TermIndex` of every term of every field before returning. It is unchanged
from the M1.6 measurement (52.7 ms vs Lucene's 0.34 ms on the merged corpus,
155x). The one improvement since is `TermIndex`'s flat `bytes` + fixed-size
`recs` layout, which removed one heap allocation per term but not the
materialization itself. No change in this batch — see B3 for why it and the
automaton work are the same milestone.

#### B4 [MISSING, low priority] `TermState` seeking

`SegmentTermsEnum.seekExact(BytesRef, TermState)`, `termState()` and `ord()`
have no counterpart. Their caller in Lucene is `TermStates` reuse across
segments in `IndexSearcher`; this port's `search/` layer does not build
`TermStates` yet (`docs/parity.md`'s `TermQuery` row says so explicitly), so
there is nothing to reuse. Recorded; it belongs with the search batches.

#### B5 [MISSING, unmeasurable here] `prefetchBlock`

`SegmentTermsEnumFrame.prefetchBlock`/`IndexInput.prefetch` (Lucene 10's
explicit `madvise(WILLNEED)`) has no equivalent. Same status as the identical
finding already recorded for `postings.rs`: it cannot be evaluated against a
warm page cache, and this port's eager open makes it moot anyway.

#### B6 [MISSING, diagnostic only] `Stats` / `FieldReader.getStats`

`Stats.java` (block counts by depth, bytes per block type) exists for
`CheckIndex` output and has no read-path role. `lucene-index/src/check_index.rs`
is explicitly scoped away from full `CheckIndex`. Recorded, not ported.

#### B7 [INTENTIONAL] `entCount == 0` is an error, not an assertion

Java `assert entCount > 0` — disabled in production, so a zero-entry block
would silently decode as empty. This port errors. Kept.

### Verdict

Two fixes (B1, B2) with tests. B3/A1 recorded with a measurement and a named
dependency chain (b8 → automata, then lazy frames) rather than half-ported;
B4/B5/B6 recorded as out-of-scope with owners. Decoding itself — trie nodes,
floor blocks, block headers, suffix compression, stats and metadata —
agrees with `lucene103/blocktree` field for field.

---

## `crates/lucene-codecs/src/terms_dict.rs`

**Java counterpart:**
`org/apache/lucene/codecs/lucene90/Lucene90DocValuesProducer.java` —
`readTermDict` and the inner `TermsDict` class — plus
`Lucene90DocValuesConsumer.addTermsDict` (writer, read to settle the
end-of-dictionary question below). This one *is* `lucene90`: the doc-values
format at 10.5.0 is still `Lucene90DocValuesFormat`.

### Method correspondence

| Rust | Java | Status |
|---|---|---|
| `read_term_dict_entry` | `Lucene90DocValuesProducer.readTermDict` | identical, field for field |
| `decode_all_terms` | `TermsDict.next()` driven to exhaustion + `decompressBlock` | same output; **T1/T2/T3 fixed**, **T4** perf |
| `read_u8` | `ByteArrayDataInput.readByte` | identical |
| `read_vint` | `DataInput.readVInt` | **was divergent — T3, fixed** |
| — | `TermsDict.seekExact(long ord)`, `seekCeil`, `seekBlock`, `seekTermsIndex`, `getTermFromIndex`, `getFirstTermFromBlock`, `ord()` | **not ported** (T5, INTENTIONAL) |

`decode_all_terms`'s "does this block have a body" test (`ord + 1 <
terms_dict_size`) differs textually from Java's (`filePointer <
termsDataLength - 1`) but is equivalent, confirmed against the *writer*:
`addTermsDict` emits a compressed body only when
`bufferedOutput.getPosition() > dictLength`, i.e. only when the block holds
more than its first term — and blocks are fixed at 64 terms, so only the last
block can be short. Checked rather than assumed.

### Findings

#### T1 [CORRECTNESS] a prefix length longer than the previous term panicked — fixed

`term.extend_from_slice(&previous[..prefix_len])` panics on corrupt input
where Java throws from its bounded `term.bytes` array. Now a `Corrupted`
error. **Test:** `decode_all_terms_rejects_prefix_longer_than_the_previous_term`.

#### T2 [CORRECTNESS] the first-term length went unchecked into `vec![0u8; n]` — fixed

A negative `vint` becomes ~2^64 as a `usize`; the infallible allocation aborts
rather than erroring. Now bounds-checked against the region. **Test:**
`decode_all_terms_rejects_first_term_length_past_the_region` (`i32::MAX`, `-1`).
Same class of defect as B2 and as `data_input.rs`'s already-documented
`readString` hardening.

#### T3 [CORRECTNESS] the block-body `vint` was unbounded — fixed

`DataInput.readVInt` throws `"Invalid vInt detected (too many bits)"` past
five bytes. This port's local `read_vint` looped, so the sixth byte shifted by
35 — a panic in debug builds, a wrong value in release. Now bounded at five
bytes. **Test:** `block_body_vint_is_bounded_at_five_bytes_like_java`, which
also pins that a legal five-byte `i32::MAX` still decodes.

#### T4 [PERF] two allocations and two copies per term — fixed

Every term was built into a `Vec`, `clone()`d into the output, and the clone
moved into `previous`; the block body was decompressed into one buffer and
then `to_vec()`'d into another to drop the dictionary prefix. Now the term is
built once and moved into `terms`, `previous` is `terms.last()`, and the block
body is `drain`ed in place — one allocation per term and one per block,
against two and two. Java has neither cost (it reuses a single `BytesRef`),
so this narrows a gap rather than opening one. Not benchmarked: the change is
strictly fewer allocations and fewer bytes copied for identical output, which
is not a case where a measurement would tell us something the diff does not.
`Vec::with_capacity(terms_dict_size)` is also now bounded by the terms
region's own length, since `termsDictSize` is untrusted.

#### T5 [INTENTIONAL] the lazy `TermsEnum` half is not ported

`seekExact(ord)`, `seekCeil`, `seekBlock`/`seekTermsIndex` and the two
`DirectMonotonicReader` arrays they need exist in Java purely for random
access without a full scan. This port materializes every term in one pass, so
they have no caller; the arrays are still *parsed* to keep the `.dvm` cursor
aligned. Already documented in the module doc, unchanged, and the same
decode-fully trade-off as `IndexedDISI` and stored fields. Worth noting it is
the same shape of trade-off as A1 — but here the dictionary is a doc-values
ordinal map, sized by distinct values of one field, not by the whole
vocabulary.

### Verdict

Swept clean. Three corrupt-input divergences fixed with tests, one allocation
cleanup, one recorded intentional scope limit. The wire-format decode itself
matched Java exactly, including the end-of-dictionary edge case.

---

## Gate

```
cargo fmt --all                                            clean
cargo clippy -p lucene-codecs --all-targets -- -D warnings  clean
cargo test -p lucene-codecs                                 828 lib + all
                                                            integration tests
                                                            pass, 0 failed
```

The new fixture was checked for the property `scripts/gen-fixtures.sh --check`
relies on: regenerating `GenFstEmptyKey` into a temp dir reproduces
`fixtures/data/fst_empty_key/` byte for byte (it writes raw FST bytes, not an
`IndexWriter` index, so it is in the deterministic set).

Not committed.
