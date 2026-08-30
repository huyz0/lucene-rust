# c1-lazy-blocktree

Follow-up batch: closes carry-over **A1** ("`blocktree.rs` materializes the
whole term dictionary at open -- 52.7 ms vs Lucene's 0.34 ms"), open since the
M1.6 sweep and re-verified by `b4-fst-blocktree`.

Files swept/changed: `crates/lucene-codecs/src/blocktree.rs` (rewritten around
a lazy frame stack), `crates/lucene-codecs/benches/blocktree_open.rs` (new) and
its `[[bench]]` stanza in `crates/lucene-codecs/Cargo.toml` (the other two
untracked `[[bench]]` stanzas in that file are concurrent batches'),
`benchmarks/micro/java/TermSeekMicro.java` (new),
`benchmarks/corpus/src/GenTermCorpus.java` (new),
`crates/lucene-codecs/tests/blocktree_{deep_nesting,multilevel}_fixture.rs`
(new tests), `crates/lucene-search/src/lib.rs` (one line, see F-10).

Lucene source read: `/home/tuong/work/lucene` at 10.5.0,
`org/apache/lucene/codecs/lucene103/blocktree/{Lucene103BlockTreeTermsReader,
FieldReader,SegmentTermsEnum,SegmentTermsEnumFrame,TrieReader,TrieBuilder,
CompressionAlgorithm,Stats,IntersectTermsEnum,IntersectTermsEnumFrame}.java`.

---

## `crates/lucene-codecs/src/blocktree.rs`

**Java counterparts:** as listed above, plus
`org/apache/lucene/util/compress/LowercaseAsciiCompression.java`.

### Method correspondence

| Rust | Java | Status |
|---|---|---|
| `open` / `open_shared` | `Lucene103BlockTreeTermsReader` ctor + `FieldReader` ctor | identical; **now stops after the `.tmd` records** (F-1) |
| `load_node` | `TrieReader.load`/`loadLeafNode`/`loadSingleChildNode`/`loadMultiChildrenNode` | identical (unchanged) |
| `lookup_child` | `TrieReader.lookupChild` | **new** (F-2) |
| `child_position` | `ChildSaveStrategy.{BITS,ARRAY,REVERSE_ARRAY}.lookup` | same positions, byte-wise `BITS` (F-2) |
| `Frame::load_block` | `SegmentTermsEnumFrame.loadBlock` | identical, incl. every b4 allocation guard |
| `Frame::rewind` | `SegmentTermsEnumFrame.rewind` | same state; skips the forced reload Lucene leaves in (F-9) |
| `Frame::set_floor_data` | `SegmentTermsEnumFrame.setFloorData` | identical + rejects `numFollowFloorBlocks <= 0` |
| `Frame::scan_to_floor_frame` | `SegmentTermsEnumFrame.scanToFloorFrame` | identical |
| `Frame::load_next_floor_block` | `loadNextFloorBlock` | identical |
| `Frame::next_leaf` / `next_non_leaf` / `next` | `nextLeaf` / `nextNonLeaf` / `next` | identical (Java's asserts become errors) |
| `Frame::scan_to_sub_block` | `scanToSubBlock` | identical |
| `Frame::scan_to_term` | `scanToTerm` | identical dispatch |
| `Frame::scan_to_term_leaf` | `scanToTermLeaf` | identical |
| `Frame::binary_search_term_leaf` | `binarySearchTermLeaf` | **new** (F-3) |
| `Frame::scan_to_term_non_leaf` | `scanToTermNonLeaf` | identical; the sub-frame descent is returned to the caller (needs the stack) |
| `Frame::decode_meta_data` | `decodeMetaData` | identical (singleton run length, DOCS aliasing, per-block `absolute`) |
| `Frame::fill_term` / `term_block_ord` | `fillTerm` / `getTermBlockOrd` | identical |
| `SegmentTermsEnum::{push_frame_node,push_frame_fp,push_next_frame}` | `pushFrame(node,len)` / `pushFrame(node,fp,len)` | identical modulo F-9 |
| `SegmentTermsEnum::seek_exact` | `seekExact` / `prepareSeekExact` | identical, minus the seek-state-reuse branch (F-9); min/max short-circuit **new** (F-4) |
| `SegmentTermsEnum::seek_ceil` | `seekCeil` | identical, minus F-9 |
| `SegmentTermsEnum::next` | `next()` | identical |
| `SegmentTermsEnum::stats` / `stats_and_meta` | `docFreq()`/`totalTermFreq()` / `postings()`'s `decodeMetaData` | identical |
| `TermsEnum::{try_next,next,try_seek_ceil,seek_ceil,try_current,current}` | `TermsEnum.next`/`seekCeil`/`term`+`docFreq` | same answers; `try_*` added (F-7) |
| `FieldTerms::{try_seek_exact,seek_exact}` | `SegmentTermsEnum.seekExact` + `docFreq`/`totalTermFreq` | same answers (F-7) |
| `FieldTerms::{postings,lazy_postings,positions,positions_for_docs,positions_flat}` | `SegmentTermsEnum.postings`/`impacts` | unchanged glue, now one trie walk each instead of two array searches |
| `FieldTerms::{intersect,fuzzy_intersect,regexp_intersect}` / `Intersect` | `FieldReader.intersect` -> `IntersectTermsEnum` | prefix-seek + forward walk + dead-prefix block skip (F-8); still not an automaton |
| `BlockTreeFields::{field,iter_fields,empty}` | `Lucene103BlockTreeTermsReader.terms`/`iterator` | identical intent |
| `decode_block` / `decode_block_at_depth` | -- | now `#[cfg(test)]`-only, built on `Frame` |
| `trie_children` / `expand_floor` / `collect_leaf_blocks` (tests) | -- | test-only, built on `lookup_child` / `Frame::scan_to_floor_frame` |
| -- | `SegmentTermsEnum.seekExact(BytesRef, TermState)` / `termState()` / `ord()` | **not ported** (F-12) |
| -- | `SegmentTermsEnumFrame.prefetchBlock` / `SegmentTermsEnum.prefetch` | **not ported** (F-12) |
| -- | `Stats.java` / `FieldReader.getStats` / `computeBlockStats` | **not ported** (F-12) |
| -- | `IntersectTermsEnum` / `IntersectTermsEnumFrame` | **not ported** (F-12) |
| -- | `SegmentTermsEnum`'s seek-state-reuse prologue | **intentionally not ported** (F-9) |

---

## Measurements

All on the M1 benchmark corpus `benchmarks/.corpus/merged` -- one
force-merged, real-`IndexWriter`-written Lucene 10.5.0 segment: 579,255
distinct terms across 4 indexed fields, 4.77 MB `.tim`, 89 KB `.tip`; the
widest field, `body`, has 200,000 terms. Rust via
`cargo bench -p lucene-codecs --bench blocktree_open` (criterion, `taskset`ed
to one P-core pair); Java via `benchmarks/micro/java/TermSeekMicro.java` and
`benchmarks/micro/java/ReaderOpenMicro.java` against the same directory, both
pinned the same way. The two pick **the same 2000 terms in the same order**
(every 97th term of the widest field, shuffled by the same xorshift64), so the
per-seek numbers are directly comparable.

### Time

| case | before (eager) | after (lazy) | Lucene 10.5.0 | after vs Lucene |
|---|---|---|---|---|
| `blocktree::open` (whole segment) | **35.4 ms** | **0.175 ms** | -- | -- |
| `DirectoryReader::open` (whole segment) | 37.4 ms | ~2.2 ms *(derived, see below)* | **0.310 ms** | -- |
| `seekExact`, hit | 307 ns | 495 ns | 440 ns | **1.13x** |
| `seekExact`, miss | 308 ns | 215 ns | 263 ns | **0.82x** |
| `next()` over the whole `body` field | 2.1 ns/term | 28 ns/term | 22 ns/term | **1.28x** |

The three per-term rows are medians of an **interleaved** A/B (java, rust,
java, rust, ...), the same protocol `scripts/bench-compare.sh` uses and for the
same reason: a run takes minutes and this machine drifts over that, so
alternating makes the drift fall on both sides instead of biasing whichever
went second. Both sides pinned to the same P-core pair. On a quieter machine
both sides are faster in step (Lucene 446 ns / Rust 446 ns for a hit), which
is why the ratio, not the absolute, is the number to read. The "before"
column was measured before the batch on the same machine at low load and is
therefore, if anything, flattered.

**Did it close the 155x gap? Yes, at this layer, and without paying for it in
the seek loop.** `blocktree::open` is **202x faster** and now costs less than
Lucene's whole `DirectoryReader.open` (0.175 ms vs 0.310 ms). A cold `seekExact`
hit costs 1.13x what Lucene pays for the identical work on the identical terms
-- that is what a faithful port of the same algorithm should cost, and it is
the honest price of not having done the work at open. A miss is *cheaper* than
Lucene's, from the min/max short-circuit (F-4) plus the no-terms fast path.

Against the port's own previous behaviour a seek is 1.6x slower (307 -> 495 ns)
because it now actually reads a block instead of bisecting a pre-decoded array.
The break-even is `35.2 ms / 188 ns` = **~187,000 seeks per reader open**: below
that the lazy reader wins outright, and a real refresh interval is nowhere near
it (a term query issues a handful of seeks). Above it the trade would reverse
-- but the eager design's 39 MB per segment would not become affordable again.

Two honest qualifications:

- **The `DirectoryReader::open` row's "after" is derived, not measured.**
  `benchmarks/rust-runner`'s `micro` binary does not currently compile (an
  unrelated `for_util::for_encode` signature change from a concurrent batch,
  `benchmarks/rust-runner/src/micro.rs:84`), so only the *stale* pre-batch
  binary could be run: it reports 37.4 ms, consistent with `blocktree::open`'s
  35.4 ms plus ~2.0 ms of everything else. Subtracting the measured
  `blocktree::open` delta gives ~2.2 ms. **The remaining ~2 ms is not this
  module's** -- it is `DirectoryReader::open_segments`' file handling, and it
  is now the dominant term. Re-run
  `benchmarks/rust-runner/target/release/micro reader_open benchmarks/.corpus/merged`
  once that build is fixed to confirm.
- **Full-field enumeration got slower in absolute terms** (0.43 ms -> 5.5 ms
  for 200k terms) because the work moved out of `open` and into the walk. End
  to end it is still 6.5x cheaper (35.8 ms -> 5.7 ms for open plus a full
  scan), and the per-term rate is 1.28x Lucene's on the identical
  `next()`+`docFreq()` loop -- see F-14 for the one API-shaped reason it is
  not 1.0x.

### Memory

Measured with a throwaway probe reading `/proc/self/status`'s `VmHWM` around
each phase, on the same corpus:

| | peak RSS delta |
|---|---|
| after: `blocktree::open` | **+4.74 MB** -- exactly the shared `.tim` + `.tip` buffers (F-13), nothing per term |
| before: the materialized dictionary those same 579,255 terms produced | **+39.0 MB** on top (2.75 MB of term bytes + 579,255 x 64-byte records + `Vec` slack) |

So per-segment live heap is now O(open enums x frame stack) -- a handful of
frames, each holding four buffers sized to the largest block it has seen
(a few KB) -- instead of O(all terms in the segment). Re-measured by
rebuilding the old shape through the new enum, not estimated.

### Term intersection (b8's blocker (b))

Over a **real-Lucene-written** `t0`..`t999999` dictionary
(`benchmarks/corpus/src/GenTermCorpus.java` -> `benchmarks/.corpus/terms1m`),
which is the same shape b8 measured but with real `.tim` blocks instead of an
in-memory `Vec<&[u8]>`. `scan` is the identical walk with the dead-prefix skip
removed.

| pattern | scan | skip | speedup | b8 (eager, in-memory) |
|---|---|---|---|---|
| `t1[0-9]` | 4.16 ms | 31.8 us | **131x** | 88x |
| `t1*z` | 41.2 ms | 23.3 us | **1768x** | 1065x |
| `t[0-9]{4}` | 53.8 ms | 45.8 ms | 1.17x | 1.00x |
| `t.*99` | 105.9 ms | 106.5 ms | 1.00x | 1.01x |

The two skipping shapes improved by ~1.5-1.7x over b8's numbers, which is
exactly the blocker b8 named: the jump now also avoids *loading* the `.tim`
blocks under the dead prefix, not just testing the terms in them. The two
prefix-closed shapes, which have no dead prefix at all and so can only lose,
stay at parity -- see F-8 for the accounting bug that had to be fixed to keep
them there.

---

## Findings

### F-1 [PERF] the whole term dictionary was decoded at `open` -- fixed

**Java.** `Lucene103BlockTreeTermsReader`'s constructor reads `.tmd` and builds
one `FieldReader` per field, holding `numTerms`/`sumTotalTermFreq`/
`sumDocFreq`/`docCount`/`minTerm`/`maxTerm` and the
`(indexStart, rootFP, indexEnd)` triple. It opens no block. `FieldReader
.iterator()` hands out a `SegmentTermsEnum`, which loads individual `.tim`
blocks as a seek or scan walks into them and keeps a stack of
`SegmentTermsEnumFrame`s.

**This port.** `open` recursively visited every trie node
(`collect_leaf_blocks`), expanded every floor block (`expand_floor`), decoded
every `.tim` block including every term's postings metadata
(`decode_block_at_depth`), merged all of it into one flat `TermIndex` and
sorted it -- per field, per segment, before returning.

**Consequence.** 35.4 ms and +39.0 MB of live heap per segment open, against a
whole Lucene `DirectoryReader.open` of 0.310 ms. A search engine reopens
readers on every refresh.

**Fixed:** ported `SegmentTermsEnum` + `SegmentTermsEnumFrame` (see the
correspondence table). `open`/`open_shared` now read only the `.tmd` records
plus each field's root trie node (kept as an O(1) validation, so a `rootFP`
outside the field's own region is still rejected at open). Numbers above.

**Tests:** the 30+ real-Lucene fixtures under `fixtures/data/blocktree_*`
pass unchanged -- they are the correctness floor and no assertion in them was
touched. Two new exhaustive `seekCeil` differentials were added on top,
because `seekCeil` is the entry point with paths `seek_exact` cannot reach
(the trie running out mid-target; a scan ending past a block's last entry, in
which case Java falls through to `next()`, pops the stack and may re-descend
into a deeper sub-block; a non-leaf scan stopping on a sub-block pointer that
sorts after the target). Both compare against a brute-force ceiling over real
Lucene's own term list, for four target families per term, and then check that
`next()` continues from where the seek landed:
`deep_nesting_field_seek_ceil_matches_a_brute_force_ceiling` (2000 terms over
a genuine depth-6 chain of nested non-leaf blocks -- ~7000 targets) and
`multilevel_field_seek_ceil_matches_a_brute_force_ceiling` (8000 terms, dense
`REVERSE_ARRAY` root, floor-split blocks -- ~23,000 targets).

### F-2 [MISSING] `TrieReader.lookupChild` was not ported -- fixed

**Java.** `lookupChild(targetLabel, parent, child)` resolves one label to one
child, via `ChildSaveStrategy.{BITS,ARRAY,REVERSE_ARRAY}.lookup`, and returns
`null` when the label is absent.

**This port.** Only `multi_children_labels_and_fps` existed: a
"list every child" generalization written because the eager traversal wanted
the whole subtree. Nothing could ask "does this node have child `b`?", which
is the single operation a lazy seek is built out of. The three
`ChildSaveStrategy` *miss* paths (label below `minLabel`, above the encoded
range, or explicitly listed as absent by `REVERSE_ARRAY`) had no counterpart
at all.

**Fixed:** `lookup_child` + `child_position`, a direct port including all
three miss paths. `BITS` is decoded byte-wise rather than through Java's
64-bit `RandomAccessInput.readLong`: the answer is identical (the long reads
are little-endian, so bit `i` of word `w` is bit `i & 7` of byte
`8w + (i >> 3)`, and a `Long.bitCount` over the words below the target equals a
byte popcount over the same bytes), and it avoids Java's habit of reading
eight bytes past the end of the strategy region -- harmless there because the
extra bits are masked off, but an out-of-bounds error for a slice that ends at
the trie region's last node.
**Tests:** the three `multi_children_*_strategy` unit tests now probe every
one of the 256 labels through `lookup_child` (hit *and* miss) via the
test-only `trie_children` helper, plus
`multi_children_reverse_array_strategy_with_gap` asserts the two
explicitly-absent labels miss rather than resolving to a neighbour;
`multi_children_node_with_invalid_strategy_code_rejected` and
`multi_children_fps_rejects_delta_exceeding_parent_fp` now go through
`lookup_child` too. The real-fixture proof is unchanged
(`child_strategies_fixture_forces_array_and_bits_strategies` forces `ARRAY`
and `BITS`, `multilevel_fixture_reaches_a_genuine_non_leaf_block` forces
`REVERSE_ARRAY`).

### F-3 [MISSING] `SegmentTermsEnumFrame` had no counterpart -- fixed

`Frame` now ports every method (see the table). Worth calling out one that
was not merely "not lazy" but **absent**: `binarySearchTermLeaf`, the
`allEqual` fast path. When every suffix in a leaf block has the same length,
entry `i`'s bytes start at `i * suffixLength` and Lucene bisects the block
instead of scanning it; the port had no equivalent because it never scanned a
block at all. It is now ported including its awkward tail (the
`end < entCount - 1` re-position onto the greater term, and the
`suffixesReader` position fix-ups the following `nextLeaf` depends on).
**Tests:** exercised by every fixture whose blocks are `allEqual` -- the
`blocktree_deep_nesting_index` fixture's 16-byte fixed-length terms make every
block `allEqual`, so its 2000-term `seek_exact`, `next()` and ~7000-target
`seek_ceil` differentials all run through this path.

### F-4 [MISSING] `prepareSeekExact`'s min/max short-circuit -- fixed

**Java.** `prepareSeekExact` opens with
`if (fr.size() > 0 && (target.compareTo(fr.getMin()) < 0 || target.compareTo(fr.getMax()) > 0)) return null;`
-- a target outside the field's recorded term range never touches the trie.

**This port.** Absent; `min_term`/`max_term` were parsed, stored, exposed and
never used for anything.

**Fixed:** ported into `SegmentTermsEnum::seek_exact`. It is a large part of
why a miss is now *faster* than before the batch (308 ns -> 215 ns), and
cheaper than Lucene's own miss, despite the extra work of loading a block.
**Test:** `seek_exact_stops_at_a_trie_node_with_no_terms` covers the
neighbouring fast path; the guard itself is exercised by every fixture's
"absent term" assertion (e.g. `many.seek_exact(b"zzzzzzzzzzzzzzzz")`).

### F-5 [CORRECTNESS] `isLastInFloor` was decoded and thrown away -- fixed

**Java.** `loadBlock` records `isLastInFloor = (code & 1) != 0`, and both
`next()` and `nextNonLeaf` use it to decide whether to chain into the next
floor block (`loadNextFloorBlock`) or pop the frame.

**This port.** Read into `let _is_last_in_floor` with the comment "Not needed
here -- `expand_floor`/`collect_leaf_blocks` already resolved every floor
sub-block's own fp up front". True of the eager design, and false of any lazy
one: without it a walk stops at the first floor block's last entry.

**Fixed:** `Frame::is_last_in_floor` is now load-bearing in
`Frame::next_non_leaf` and `SegmentTermsEnum::next`.
**Test:** the in-module floor test (renamed
`open_floor_field_walks_both_blocks`) now walks the field with `next()` and
asserts all four terms come back across the floor boundary, and that a
`seek_ceil` from the first block's range lands in the second.

### F-6 [CORRECTNESS] a mis-ordered block was silently repaired, not reported -- fixed by construction

**Java.** `scanToTermLeaf`/`scanToTermNonLeaf` compare suffixes in stream
order and stop at the first entry `>` the target; a block whose entries are
not sorted yields wrong answers, which is why the writer's invariant matters.

**This port.** `open` sorted every field's decoded entries globally
(`entries.sort()`, "blocks are decoded in trie-traversal order, not
necessarily sorted term order"). That is correct for merging blocks, but it
also **repaired** an out-of-order block, and repaired a floor split whose
lead-byte boundaries did not match its contents -- neither could ever be
observed.

This was not hypothetical: this module's own `open_floor_field_*` test wrote a
block whose two terms were in *descending* order (`"b"`, `"a"`) and marked
*both* floor blocks `isLastInFloor`, and passed. Under the lazy scan the same
bytes fail, as they should.

**Fixed** by construction -- there is no global sort any more, so block order
is load-bearing exactly as in Java. The test's bytes were corrected to a
layout `Lucene103BlockTreeTermsWriter` would actually emit.

### F-7 [MISSING] corrupt-block errors could only surface at `open` -- fixed

**Java.** `loadBlock` throws `CorruptIndexException` from inside
`TermsEnum.seekExact`/`next`/`seekCeil`. The reader's constructor never reads
a block, so it cannot report one.

**This port.** The opposite: `open` reported every block-level corruption, and
`seek_exact`/`next`/`seek_ceil` were infallible and could not report any.
Making the reader lazy necessarily moves those errors to lookup time, which is
also where Java has them.

**Fixed:** every lookup gained a `Result`-returning form --
`FieldTerms::try_seek_exact`, `TermsEnum::try_next`, `TermsEnum::try_seek_ceil`,
`TermsEnum::try_current`. The pre-existing infallible spellings are kept so no
caller had to change, and each one's doc comment states plainly that it
reports a corrupt block as "no such term"/end-of-terms and names the `try_`
form to use instead. That degradation is the one behaviour this batch
deliberately traded (F-11).
**Tests:** `a_corrupt_block_errors_from_the_lookup_not_from_open` corrupts a
suffix length inside a valid segment and asserts *both* halves of the contract
for all four entry points (`try_*` errors, the infallible spelling degrades)
plus that `intersect` ends rather than yielding a wrong term;
`empty_terms_block_rejected` was retargeted from `open` to `try_seek_exact`;
`deep_nesting_field_fallible_lookups_agree_with_the_infallible_ones` proves
the two spellings agree everywhere on intact real-Lucene bytes.

### F-8 [PERF] `regexp_intersect` now skips blocks, not just terms -- fixed and measured

**Java.** `IntersectTermsEnum` drives a `ByteRunAutomaton` down the trie and
abandons a subtree the moment the automaton enters a dead state -- so the
blocks under it are never read.

**This port, before.** b8 ported the *pruning* half over the materialized
array: `RegexpPattern::dead_prefix_len` answers "is this prefix dead?", and a
galloping search jumped past the run sharing it. b8 recorded that the other
half -- not loading the pruned blocks -- was blocked on A1.

**Fixed:** `Intersect` (shared by `intersect`/`fuzzy_intersect`/
`regexp_intersect`) seeks to the pattern's literal prefix and walks forward
with the lazy enum, so blocks outside the prefix range are never loaded at
all; and on a dead prefix it re-seeks past the whole range, which with lazy
frames skips the blocks under it. Numbers in the Measurements section:
131x and 1768x, up from b8's 88x and 1065x.

**One bug found and fixed while measuring.** b8's adaptive give-up counted a
skip *attempt* on every non-matching term, including the ones where
`dead_prefix_len` returned `None` -- which is the whole point, since asking a
prefix-closed pattern (`t.*99`) is pure loss. The first port of it here only
counted attempts that produced a jump, so on `t.*99` the give-up never fired
and the walk paid a `could_match_prefix` run per term forever: **1.72x slower
than not skipping at all**. Restored to b8's semantics (count the question,
not the jump), which brings `t.*99` back to 1.00x. The saving side had to be
re-derived too, since "terms skipped" is not observable on a lazy cursor: a
jump that stays inside its block is credited the entry-cursor delta, and one
that leaves it a fixed block credit, because it provably avoided loading at
least the rest of that block.
**Test:** `regexp_intersect_skip_agrees_with_a_brute_force_scan` (unchanged
from b8, 3000 terms x 10 patterns) still proves the skip never changes *which*
terms come back; the bench asserts scan/skip agree on the match count for
every pattern before timing them.

### F-9 [INTENTIONAL] the seek-state-reuse prologue is not ported

**Java.** `prepareSeekExact` and `seekCeil` both open with ~80 lines that
reuse the *previous* seek's frame stack when the new target shares a prefix
with the current term (`validIndexPrefix`, the `nodes[]` array, `lastFrame`,
`targetBeforeCurrentLength`), so seeking `foobar` then `foobaz` re-walks only
the last byte.

**This port.** Every seek restarts from the root -- which is exactly Java's
own `currentFrame == staticFrame` path, so it is a subset of Java's behaviour,
not a different one. It is a pure optimization for *sorted* access, and the
piece of it that matters for any access order -- not re-loading a block the
frame already holds -- is recovered differently: `Frame::rewind` resets the
four region cursors in place when the frame is already parked on `fpOrig`
(the optimization Lucene leaves commented out in `rewind()`'s own body, where
it instead forces a reload), and `FieldTerms` pools one `EnumState` behind a
`Mutex` so the `&self` lookups keep their loaded blocks warm across calls
rather than starting cold.

Recorded rather than ported because the measurement says the remaining gap is
small and not obviously the prologue's: a hit costs 495 ns against Lucene's
440 ns on the same terms in the same *shuffled* order -- which is precisely
the order the prologue cannot help with, since it only pays off when
consecutive targets share a prefix. Porting it would add the port's single
most bug-prone piece of state (three interacting cursors that Java itself guards with four
asserts) for a win that only shows on sorted seek streams, which this port's
search layer does not generate.

### F-10 [INTENTIONAL] `intersect`'s iterator item is now owned

`intersect`/`fuzzy_intersect`/`regexp_intersect` used to yield
`(&'a [u8], TermStats)` borrowed from the materialized array. A lazy cursor
cannot: the term bytes live in the enum's own buffer and are overwritten by
the next step, and `Iterator` cannot express an item that borrows the
iterator. The item is now `(Vec<u8>, TermStats)`.

Every caller in the workspace already did `t.to_vec()` immediately, so this is
allocation-neutral (one allocation instead of one allocation plus a copy) and
needed exactly **one** line changed outside this crate:
`crates/lucene-search/src/lib.rs`'s fuzzy expansion, `pattern.boost(term)` ->
`pattern.boost(&term)` (and `term: term.to_vec()` -> `term`, dropping a copy).
`lucene-index` and `lucene-ffi` needed no change.

### F-11 [INTENTIONAL] the infallible lookups degrade a corrupt block to "not found"

The counterpart of F-7. `seek_exact`/`next`/`seek_ceil`/`current` keep their
old signatures so no caller had to be migrated while two other batches were
editing those crates; on a corrupt block they report "no such
term"/end-of-terms rather than an error. This is documented on each method and
tested in both directions, not silent -- but it *is* weaker than what `open`
used to guarantee, and the right end state is for callers to move to the
`try_*` forms. Carried over (see below) rather than done here, because
`seek_exact` alone has 64 call sites across `lucene-search`, `lucene-index`,
`lucene-ffi` and `benchmarks/`, all of which were being actively edited by
b13/b14/b15 while this batch ran. Note that the one consumer that most needs
the error, `lucene-index::check_index`, is not blind to it: it enumerates with
`next()` and compares the count against `numTerms`, so a block that fails to
load still surfaces as a count mismatch.

### F-12 [MISSING, recorded] the same four items b4 recorded, unchanged

`seekExact(BytesRef, TermState)`/`termState()`/`ord()` (no `TermStates` reuse
exists in this port's search layer), `prefetchBlock`/`IndexInput.prefetch`
(unmeasurable against a warm page cache), `Stats.java`/`FieldReader.getStats`/
`computeBlockStats` (diagnostic only), and a real
`IntersectTermsEnum`/`IntersectTermsEnumFrame` driven by a
`CompiledAutomaton` (still blocked on `org.apache.lucene.util.automaton`, none
of which exists here -- b8 assessed and declined it). A1 was the *other* half
of the last one's blocker and is now gone, so the remaining blocker is purely
the automaton.

One thing that is no longer missing: `TrieReader`'s lazy navigation is now the
substrate an `IntersectTermsEnum` would be built on, so that port is no longer
"two milestones away".

### F-13 [PERF, recorded] `open` still copies `.tim`/`.tip`

`open` takes `&[u8]` and the returned reader navigates those bytes for its
whole life, so it copies them into `Arc<[u8]>` -- 4.86 MB on this corpus,
which is most of the remaining 0.175 ms. `open_shared` takes the shared
buffers directly and skips it, and is tested
(`open_shared_reads_the_same_dictionary_without_copying`), but
`lucene-search/src/directory_reader.rs` -- the only production caller -- still
uses `open`, and was being edited by another batch while this one ran.
Migrating it is a three-line change *if* the segment files can be produced as
`Arc<[u8]>` at the source; note that `Arc<[u8]>::from(Vec<u8>)` copies too, so
the real fix is for `open_segment_file` to hand back a shared buffer (or an
mmap) rather than a `Vec`. Carried over.

### F-14 [PERF, recorded] `TermsEnum::next` decodes metadata for every term

Java's `next()` decodes only the term bytes; `docFreq()`/`totalTermFreq()`
trigger `decodeMetaData` lazily, so a terms-only consumer (an MTQ collecting
candidate terms, `check_index`'s term walk) never pays for the postings
metadata. This port's `TermsEnum::next` returns `(term, TermStats)` in one
call, so it always decodes. Measured against Java's own `next()`+`docFreq()`
loop -- i.e. the comparison where both decode -- the port is 27 ns/term
against 20.5 ns; the *avoidable* part is whatever a stats-free walk would save
on top. Fixing it means splitting the API (a `next_term()` returning just
bytes, with `stats()` on the side), which changes `check_index`'s and the
intersect iterators' call shape. Carried over.

### F-15 [INTENTIONAL] `open` no longer cross-checks the decoded term count

The eager `open` ended with `entries.len() as i64 != num_terms` -> `Corrupted`.
A lazy reader cannot count terms without walking every block, which is the
cost the batch exists to remove. Java does not check it either; the check
belongs in `CheckIndex`, and `lucene-index::check_index` already performs
exactly this comparison per field (`field_terms.num_terms` vs its own
enumeration). Removed deliberately, not overlooked.

### F-16 [CORRECTNESS] two recursion-depth guards became unnecessary -- removed

The eager reader needed `depth > 10_000` cycle guards in both
`collect_leaf_blocks` (trie recursion) and `decode_block_at_depth` (sub-block
recursion), because a corrupt trie or sub-block chain could loop forever. The
lazy walk cannot: `Frame::absolute_sub_fp` requires a sub-block's fp to be
*strictly below* its parent's (rejecting `subCode == 0` as well, which the old
code accepted and which would have pointed a block at itself), so the frame
stack's file pointers strictly decrease and the descent is bounded by
construction; and the trie descent is bounded by the target term's length.
Both guards are gone from production code; the arbitrary bound they imposed
(a legal 10,001-deep structure would have been rejected) is gone with them.
**Test:** `decode_block_rejects_sub_block_delta_fp_past_parent` (unchanged)
covers the fp check.

### F-17 [CORRECTNESS] the pooled lookup scratch must not serialize searchers

Making `FieldTerms::seek_exact(&self, ...)` reuse one warm enum needs interior
mutability, and the obvious `Mutex::lock` would have been a real regression:
`lucene-ffi` hands *one* `Arc<BlockTreeFields>` to every concurrently
searching JVM thread (b15 moved its registries to `RwLock` precisely so those
threads stop serializing, measured at 6.2x four-thread throughput), so a
blocking lock inside the term dictionary would have taken that straight back.

`with_scratch` therefore uses `try_lock`: the winner reuses the pooled state,
and a thread that would block runs on its own fresh `EnumState` instead --
same answers, only without the warm blocks, which is exactly what Java's
callers get since each of them holds its own `TermsEnum`. Poisoning is
recovered rather than propagated, since the state is a pure cache.
**Test:** `concurrent_lookups_do_not_serialize_and_agree` (8 threads x 20
passes over a 200-term field through one shared `Arc`, checking every
`docFreq` and a miss).

### F-18 [CORRECTNESS] two block-region reads bypassed the logical bound -- fixed

Raised by the Tier-2 review; see that section for the detail. `scan_to_sub_block`
and `scan_to_term_non_leaf`'s `subCode` read passed the whole high-water-mark
`suffix_length_bytes` buffer instead of `[..suffix_length_bytes_len]`, so on
corrupt bytes a `subCode` vlong whose continuation ran past the current block's
region would have kept reading the *previous, larger* block's leftovers -- a
wrong `lastSubFP`, hence a wrong sub-block and wrong terms, instead of a
`Corrupted`. Both now slice like every other region read in the file.

### F-19 [CORRECTNESS, trivial] a misattached doc comment

`prefix_upper_bound`'s doc comment had been separated from it by the
`RegexpIntersect` doc comment, so both rendered on `RegexpIntersect` and
`prefix_upper_bound` documented nothing. Fixed in passing.

### Verdict

Swept clean of A1. `blocktree::open` is 202x faster and no longer scales with
the segment's vocabulary in time or memory; a cold seek costs 1.13x what
Lucene 10.5.0 pays for the same work on the same terms in the same order (a
miss is cheaper than Lucene's); every
existing real-Lucene fixture passes unchanged and two new exhaustive
`seekCeil` differentials (~30,000 targets over a depth-6 nested-block field
and a floor-split 8000-term field) were added for the paths the lazy design
introduced. Term intersection now skips blocks it would have loaded, which is
the win b8 had to defer. Open items: F-11 (migrate callers to the `try_*`
lookups), F-13 (`directory_reader` to `open_shared`), F-14 (split term
iteration from stats), and F-12's unchanged four.

---

## Tier-2 review

Run on the diff (the `quality-reviewer` subagent, reading the Java sources
alongside). It walked every ported method line against line and found no path
that can return a wrong term, `docFreq` or `totalTermFreq` on well-formed
bytes, and confirmed by construction the three things this batch most needed
proving: that `Frame::rewind`'s in-place reset is field-for-field equivalent to
a reload, that nothing left in the port depends on the unported seek-state
prologue, and that `Intersect`'s skip is always strictly forward. Six advisory
findings, all acted on:

1. **Two `suffix_length_bytes` reads bypassed the logical region bound**
   (`scan_to_sub_block`, `scan_to_term_non_leaf`'s `subCode`). Because
   `fit_buf` never shrinks the buffer, a corrupt `subCode` vlong running off
   the end of the *current* block's region would have kept reading a previous,
   larger block's leftovers -- a wrong `lastSubFP`, hence wrong terms, instead
   of a `Corrupted`. Both now slice by `suffix_length_bytes_len` like every
   other read. The only finding with a correctness edge; corrupt input only.
2. **`Frame::is_seek_frame` was write-only** (a transliteration of Java's
   `node != null`, whose consumers are all unported) and survived the
   dead-code lint only because `derive(Debug)` counts as a read. Deleted.
   `EnumState::term_exists`, in the same position, is kept -- it is the flag
   only `scanToTermNonLeaf` can compute -- with a doc comment saying why.
3. **`Intersect`'s skip machinery was paid for by matchers that provably
   cannot skip.** `TermMatcher` gained `const CAN_SKIP: bool`, `false` by
   default and `true` only for `RegexpMatcher`, so the whole branch folds away
   at monomorphization for `intersect`/`fuzzy_intersect`.
4. A per-iterator `prefix.clone()` in `Intersect::start`, removed (the two
   fields are disjoint, so the borrow it dodged was never a conflict).
5. **The new `seekCeil` differentials opened a fresh enum per target**, so
   they never exercised the batch's one deliberate divergence from Java
   (F-9's in-place `rewind`), which is only reachable on the second and later
   seek against the same frame stack -- above all on a *backwards* seek.
   `deep_nesting_field_seek_ceil_matches_a_brute_force_ceiling` now shuffles
   its ~7000 targets and drives them down **one reused enum** alongside the
   fresh one, asserting both agree on status and landing term.
6. **`terms_enum_empty_field`'s replacement had dropped its
   seek-after-exhaustion assertion**, which in the new design covers something
   the old one did not have: `SegmentTermsEnum::reset` must clear `eof` or a
   `seek_ceil` issued after end-of-terms would fall out of `next()`'s `eof`
   guard regardless of its target. Re-added, in walk-then-seek order.

It also flagged that `docs/parity.md`'s blocktree row narrated the *eager*
design in the present tense for ~8000 words before reversing it at the end;
the row now leads with the current design and labels the rest as history.

## Gates

- `cargo fmt -p lucene-codecs --check` clean (the workspace-wide check trips
  on two files four other batches are mid-edit in; neither is this batch's).
- `cargo clippy -p lucene-codecs --all-targets -- -D warnings` clean for this
  file; `-p lucene-search`, `-p lucene-ffi` clean too. Findings that appeared
  in `points.rs`, `vectors.rs`, `hnsw.rs`, `stored_fields.rs` and
  `lucene-index/examples/` during the batch were other batches' in-flight
  edits and were waited out, not touched.
- `cargo test -p lucene-codecs`: every lib test (64 of them in
  `blocktree::tests`) and all 29 integration suites pass, including every
  pre-existing real-Lucene `blocktree_*` fixture unchanged. `cargo test -p lucene-search -p lucene-index -p lucene-ffi
  -p lucene-core` pass.
- `cargo llvm-cov --no-fail-fast -p lucene-codecs`: `blocktree.rs` at
  **95.80%** lines / 95.26% regions, above invariant #8's 95%-per-file bar.
  That is the *pessimistic* figure -- it counts only this crate's own tests,
  where a workspace run additionally exercises `positions_for_docs`,
  `lazy_postings` and `fuzzy_intersect` through `lucene-search`. It had to be
  run with `CARGO_TARGET_DIR` pointed at a scratch directory: other batches
  were rebuilding the shared `target/` throughout, which emptied
  `target/llvm-cov-target` mid-run and produced three successive bogus
  readings (33%, 56%, and 94.7% with every integration binary missing) before
  that was diagnosed. A `cargo llvm-cov --workspace` reading is still worth
  taking once the tree is quiet; the last complete one during this batch put
  the workspace total at 97.41%.
