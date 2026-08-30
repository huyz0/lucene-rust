# c38 — allocation shape: the payload slot, the streamed terms dictionary, and the block pool that is a milestone

Batch scope: the three largest remaining memory-shape divergences from Java,
`LEDGER.md` open-work items **15** (`indexing_chain` allocates per
token/term/posting), **16** (payload slots cost the slot, not the bytes) and
**17** (`OrdinalMap::build` materializes every segment's term list).

**Outcome: 16 and 17 closed; 15 measured, sized and declined as a milestone.**
Seven findings — 0 CORRECTNESS, 0 MISSING, 6 PERF (5 fixed and measured, 1
assessed and stopped), 1 INTENTIONAL. `verify-write-path.sh` **23/23**,
`scripts/docker-test.sh gate` green, no file below 95% lines.

## How everything here was measured

Timing on this host drifts by ~30% between windows — the "before" arm of one
payload measurement read 41.6 µs/doc in one window and 54.3 µs/doc an hour
later — so **no number below is a remembered one, and no two numbers are
compared across windows.** Every throughput figure is an **alternating
min-of-N A/B between two binaries built from the two trees**, run inside
`scripts/docker-test.sh`:

1. `git stash` the batch, build `index-bench`, copy it aside as
   `index-bench-before`;
2. `git stash pop`, rebuild, copy aside as `index-bench-after`;
3. for `i in 1..N`, for each arm, run *before* then *after* back to back into
   fresh output directories, and keep the minimum of each.

`writer_peak_kb` (`VmHWM` minus the RSS sampled after the corpus is built) is
minned the same way. This is c24's method — that batch found criterion
reporting 83/91/129 µs for identical code — and it is what makes a +2.5%
reading in one window and a −5.6% reading in another resolvable: they are the
same measurement, and running the two arms adjacently is the only thing that
makes either trustworthy.

Two instruments, both committed:

- **`crates/lucene-index/examples/invert_memory.rs`** (new). Rebuilds
  `index-bench`'s corpus exactly (20 000 documents × 40 tokens from a 20 000-word
  vocabulary), inverts it, and prints `InMemoryInvertedIndex::ram_bytes_used()`
  — an exact count of the structure, not a sample — against the body text it
  came from, with the RSS delta beside it and a breakdown by what holds the
  bytes (which sums to the total, with the remainder printed, so the columns
  cannot look complete while being a subset). This is item 15's instrument.
  Because it is new and untracked, `git stash` does not hide it: the
  **before** arm was produced by building it against the stashed tree with its
  three `TermPostingList` accessors spelled the old way (`Vec::len` on the map
  value) — a four-line edit that changes nothing it measures — and both
  binaries were then kept side by side and run alternately, like the
  `index-bench` pair.
- **`crates/lucene-search/examples/ordinal_map_memory.rs`** (c29's, reworked).
  Now takes the arm (`materialized` / `streaming`) as an argument and runs
  **one arm per process**, because measuring the second arm's peak in a
  process that has already freed 267 MB measures the allocator, not the arm.
  It also reports a total (terms produced + map built), which is the only
  figure the two arms can honestly be compared on: the materialized arm's
  `build` excludes producing the term lists, and the streaming arm's includes
  it.

---

## 1. `crates/lucene-index/src/indexing_chain.rs` — the block pool

Java: `index/IndexingChain.java`, `index/TermsHash.java`,
`index/TermsHashPerField.java`, `index/FreqProxTermsWriterPerField.java`,
`index/FreqProxFields.java`, `index/ByteSlicePool.java`,
`index/ByteSliceReader.java`, `index/ParallelPostingsArray.java`,
`util/ByteBlockPool.java`, `util/IntBlockPool.java`, `util/BytesRefHash.java`.

### Finding 1 `[PERF — measured, sized, declined: this is a milestone]`

**Java**: inverting a document costs **no per-term and no per-occurrence heap
object**. `BytesRefHash` maps term bytes to a dense term id with the bytes
living in a `ByteBlockPool`; `IntBlockPool` holds each term's two stream
addresses; `ByteSlicePool` holds the freq and prox streams as slice chains
that grow through a fixed tier ladder. `TermsHashPerField.add` dispatches each
token into `FreqProxTermsWriterPerField.newTerm`/`addTerm`, which write vints
into those slices through `TermsHashPerField.writeVInt` as the tokens arrive,
and `FreqProxFields` presents the whole thing back as a `Fields`/`TermsEnum`
at flush.

**We do**: `invert_documents_with_payloads` builds
`BTreeMap<(String, String), TermPostingList>`, one `Vec<PostingEntry>` per
term and one `Vec<Occurrence>` per posting entry.

**Measured, re-run in this session** (`invert_memory`, 20k × 40 tokens,
4.90 MB of body text, 20 000 distinct terms, 799 250 posting entries,
800 000 occurrences):

| | `InMemoryInvertedIndex` | × text | RSS delta | of which entry slots | of which occurrence vectors |
|---|---|---|---|---|---|
| before c38 | **102.48 MB** | 20.9x | 118.3 MB | — | — |
| after c38 (finding 2) | **75.82 MB** | 15.5x | 90.8 MB | **36.77 MB** | **36.59 MB** |

(The remaining 2.47 MB is the `(field, term)` key slots and their bytes; the
example prints the unaccounted remainder, which is zero.)

(c3 quoted 9.4x against 8.3 MB of *document* text, which counts the stored
`id` field too; this instrument measures the indexed field's text, which is
why the ratio differs. The MB column is the comparable one, and c3's 78.5 MB
plus c23's 24-bytes-per-entry payload slot is the 102.5 MB this batch started
from.)

**97% of what is left is two things**: the `Vec<PostingEntry>` slots
(36.77 MB) and the `Vec<Occurrence>` capacity-4 first allocations (36.59 MB —
48 bytes of allocation holding 12 bytes, on a corpus where nearly every
`(doc, term)` pair is unique). Those are exactly the two allocations a
slice-chained byte pool removes, and nothing short of the pool removes them:
c3 already tried `shrink_to_fit` and measured it costing 25–60% throughput
while peak RSS did not move at all, because glibc keeps the freed 48-byte
chunks in its arena.

**Why it is a milestone and not a batch.** Three reasons, in increasing order
of how much they cost:

1. **The pool itself is ten Java classes and 2 800 lines** (counted in the
   pinned tree), none of which exists here in any form. At this port's
   ≥95%-per-file coverage bar that is not a 2 800-line change.
2. **The output type is consumed three times.** `index_writer` reads
   `InMemoryInvertedIndex` in `build_postings_output`, `build_norms_output`
   and `build_term_vectors_output`. A pooled representation is not a `BTreeMap`
   and cannot be iterated as one; all three move onto a `TermsEnum`-shaped
   read-back (`FreqProxFields`), which is another 546 lines of Java before the
   consumers are rewritten.
3. **The win only fully lands with a borrowed-token analyzer API, which is
   another crate's.** `Analyzer::analyze` returns `Vec<Token>` where
   `Token { term: String, … }` — so **a heap allocation per token already
   happened before the indexing chain sees anything**, and a pool underneath it
   would remove the per-term and per-occurrence objects while leaving the
   per-token one. Closing that is `lucene-analysis`'s API contract, not this
   module's, and it is what c3 named when it wrote "needs a borrowed-token
   `Analyzer` API and a byte-pool posting representation".

**Resolution: stopped, as the brief directs.** The measurement, the
instrument and the 36.8 + 36.6 MB target are recorded on
`InMemoryInvertedIndex::ram_bytes_used`'s doc comment and in `LEDGER.md`
item 15 so the milestone can be planned against a number.

What this batch *did* take out of the same structure is finding 2, which is
26.7 MB of the 102.5 — and 19 MB of that is paid by every field, including the
overwhelming majority that have no payloads at all.

---

## 2. Payloads: a flat run per term, on both sides of the crate boundary

Files: `crates/lucene-index/src/indexing_chain.rs`,
`crates/lucene-index/src/index_writer.rs`,
`crates/lucene-index/src/merge.rs`,
`crates/lucene-codecs/src/postings_writer.rs`.

Java: `index/FreqProxTermsWriterPerField.writeProx`,
`codecs/lucene104/Lucene104PostingsWriter.{addPosition, startDoc, finishTerm}`.

| Rust | Java | verdict |
|---|---|---|
| `TermPostingList` (new) | the per-term slice chain `FreqProxTermsWriterPerField` writes into | new type, same role |
| `TermPostingList::{payload_bytes, payload_lengths}` | `Lucene104PostingsWriter.{payloadBytes, payloadLengthBuffer}` | identical layout |
| `TermPostingList::payload(entry, occurrence)` | *(none — Java never needs random access)* | not-in-Java; today only the tests that assert the run means what it says |
| `TermPostingList::sort_by_doc_id` | *(none — Java's postings arrive doc-ordered by construction)* | not-in-Java |
| `PostingEntry::payloads` | — | **removed** |
| `TermPostings::{payload_bytes, payload_lengths}` | `payloadBytes`/`payloadLengthBuffer` | identical layout |
| `postings_writer::permute_payload_run` | *(none)* | not-in-Java; shared by the merge and the invert |
| `write_position_tail` | `Lucene104PostingsWriter.finishTerm`'s tail | unchanged behaviour, now borrows instead of flattening |

### Finding 2 `[PERF → fixed]` the payload slot, and the 24 bytes every field paid for it

**Java**: `FreqProxTermsWriterPerField.writeProx` appends the payload length as a vint and its bytes into
the term's existing byte-slice stream. A payload costs bytes in a pool; it
costs no object, and a field with no payloads costs nothing at all.

**We did**: `PostingEntry::payloads: Vec<Vec<u8>>` — a vector header on
*every* posting entry (24 bytes, whether or not the field has payloads) plus
one allocation per posting entry plus one per non-empty payload; and
`TermPostings::payloads: Vec<Vec<Vec<u8>>>`, a second 24-byte header per entry
in the copy `build_postings_output` builds. c23 measured the result at 26 µs/doc
and ~190 MB per 50 000 documents, and identified the cause correctly with an
all-empty-payload control that cost the same.

**Fixed**, and in one place c23 did not specify: the run is held **once per
`(field, term)`**, not once per posting entry.

- `InMemoryInvertedIndex::terms` is now
  `BTreeMap<TermKey, TermPostingList>`, where `TermPostingList` is the
  `Vec<PostingEntry>` plus `payload_bytes: Vec<u8>` and
  `payload_lengths: Vec<u32>` — every occurrence's payload concatenated in
  entry-then-occurrence order, one length each.
- **`PostingEntry` has no payload field at all**, dropping from 56 to 32 bytes
  on a 64-bit target. That is the change that pays back on the ~all fields
  that have no payloads: at ~1.20 M entry slots *at capacity* it is
  essentially the whole 26.7 MB, less the ~1 MB `TermPostingList` costs by
  widening each map value from 24 to 72 bytes.
- `TermPostings` takes the identical pair, so `build_postings_output` moves
  the run across with `mem::take` — no copy, no re-materialization — and
  `write_position_tail` **borrows** it. That function already flattened the
  nested form into exactly `(payload_bytes, payload_lengths)` internally
  before doing anything with it; the flattening loop is simply gone.
- The per-document grouping map accumulates into the same flat pair, so a
  group costs two transient allocations instead of one plus one per non-empty
  payload, and — because the group's run is appended to the term's run and
  then dropped — **nothing per occurrence stays live until the flush**.
- `merge.rs` builds the run directly from `Position::payload` rather than a
  `Vec<Vec<Vec<u8>>>` it would have to flatten again.

Doing only half of it would have been slower, exactly as c23 predicted; doing
it at the entry level as a plain `(Vec<u8>, Vec<u32>)` pair would have been
*worse than the status quo* for the no-payload case (48 bytes per entry
instead of 24), which is why the run moved up to the term.

**Measured** — alternating min-of-N A/B between two binaries, 50 000
documents, four `LUCENE_RUST_INDEX_OPTIONS` arms. Two independent windows are
given because that is the honest way to report a machine that drifts: the
second was run after the Tier-2 review's fixes, in a much quieter window
(`freqs` at 19.1 µs/doc, matching c3's own baseline for this shape).

| arm | µs/doc, window A (min-of-10) | µs/doc, window B (min-of-8) | writer peak RSS |
|---|---|---|---|
| `freqs` | 28.50 → 25.27 (**−11.3%**) | 19.32 → 19.07 (−1.3%) | 156.8 → 127.7 MB (**−18.6%**) |
| `offsets` | 33.93 → 31.40 (−7.5%) | 24.99 → 23.98 (−4.0%) | 165.1 → 148.8 MB (−9.9%) |
| `payloads` | 49.94 → 47.12 (−5.6%) | 34.78 → 31.76 (−8.7%) | 342.3 → 174.1 MB (**−49.1%**) |
| `payloads-empty` | 53.80 → 40.63 (**−24.5%**) | 32.92 → 27.63 (**−16.1%**) | 303.4 → 163.6 MB (**−46.1%**) |

The memory column is stable to three digits across both windows, which is what
one expects of a `VmHWM` high-water mark and is why it is quoted once. The
time column is not stable in *magnitude* — `freqs` reads −11.3% in the loaded
window and −1.3% in the quiet one — but it is stable in **sign**: every arm,
in both windows, is faster or level. That is the claim being made, and it is
the one the brief asks for: **no arm trades throughput for memory.** (The
magnitude difference is itself informative — the busier the machine, the more
the removed allocations are worth.)

The `freqs` row is the one worth reading twice: an arm with no payloads
anywhere got 19% smaller, because it had been paying 24 bytes per posting
entry for a vector it never used.

An earlier build of this change measured `payloads-empty` at **+8.3%** time.
The cause was a repair pass in `build_postings_output` that summed every
payload length on every flush to make the "one length per occurrence"
guarantee load-bearing rather than assumed. It now runs only when the
lengths and the occurrence count actually disagree — the guarantee is still
enforced, the O(occurrences) walk is not paid to confirm it. That is recorded
because "the defensive pass costs 8%" is the kind of thing that reads as
noise if the A/B is not tight enough to see it.

**Tests.** Shape rather than timing, as the brief directs:
`a_posting_entry_carries_no_payload_slot` (asserted against `size_of`, so it
is a statement about the type and holds on a 32-bit target);
`an_all_empty_payload_field_costs_only_a_length_per_occurrence` (c23's finding
from the other side — the extra cost of declaring payloads on a field that
carries none must be *cheaper than one empty vector header per occurrence*,
which is precisely what the old shape charged before a single byte was
stored). Behaviour:
`sorting_entries_by_doc_id_carries_the_payload_run_with_them` drives three
documents in descending doc-ID order whose occurrence counts *and* payload
lengths all differ, so neither a permutation applied to only half the state
nor a fixed-width slicing bug can pass; `permute_payload_run`'s two tests pin
the permutation direction with a 3-cycle and the saturation on a truncated
run. Codec-side: `rejects_payload_bytes_that_do_not_match_their_lengths` and
`rejects_a_payload_length_that_does_not_fit_an_i32` are validation the nested
shape could not even express (a `Vec<u8>` is its own length).

**The bytes on disk are unchanged**, which is the load-bearing negative
result: `scripts/verify-write-path.sh` is **23/23**, including
`VerifyPositionsSegment`, which walks 51 documents of a six-field index with a
fresh `PostingsEnum` each and compares every occurrence's payload against what
real Lucene 10.5.0 decodes.

### Finding 3 `[PERF → fixed]` two copies of one permutation

While flattening: `merge.rs` and the new `sort_by_doc_id` both needed "reorder
a flat payload run to follow a document permutation", and the run carries no
per-document index, so both needed the same prefix-sum over the
pre-permutation occurrence counts. They are one function,
`postings_writer::permute_payload_run`, living beside the representation it
reorders rather than duplicated in the two crates that build it. It also
removed four `#[allow(clippy::arithmetic_side_effects)]` sites, because the
shared version saturates rather than indexing.

### Finding 4 `[INTENTIONAL]` the group-level run still costs two allocations

Within one document, occurrences are grouped by term before they become a
posting entry, so a payload group still allocates its own `Vec<u8>` and
`Vec<u32>` — freed as soon as the group is appended to the term's run.
Removing those means not grouping per document, which is finding 1's pooled
representation, not this batch's. Recorded rather than half-fixed: they are
transient, they no longer survive to the flush, and they are strictly fewer
than the `1 + one-per-non-empty-payload` they replaced.

---

## 3. `crates/lucene-codecs/src/terms_dict.rs` — the cursor three batches asked for

Java: `codecs/lucene90/Lucene90DocValuesProducer.TermsDict`.

| Rust | Java | verdict |
|---|---|---|
| `TermsCursor::open` | `TermsDict`'s constructor | added |
| `TermsCursor::next_term` | `TermsDict.next()` | added, identical decode |
| `TermsCursor::read_block_first_term` | `TermsDict.decompressBlock` + the uncompressed first term | added (extracted) |
| `TermsCursor::read_prefix_compressed_term` | `TermsDict.next()`'s prefix/suffix branch | added (extracted) |
| `decode_all_terms` | *(no Java counterpart — Java has no materializer)* | **now the cursor collected** |
| `read_term_dict_entry` | `readTermDict` | unchanged |
| `TermsDict.seekExact`/`seekCeil`/`ord` | — | still unported (no caller; recorded on the module doc, unchanged from b4) |

### Finding 5 `[PERF → fixed]` "needs a `TermsCursor`, which does not exist"

**Java**: `TermsDict` is an enumerator. `next()` decodes one term into a
reused `BytesRefBuilder`, taking the prefix it keeps from the term already in
that buffer.

**We did**: `decode_all_terms` and nothing else — the whole dictionary or
none of it. Three separate ledger entries (c12's Tier-2 review, c29 §6.1,
open-work item 17) named the absence of a cursor as their blocker.

**Fixed**, and the shape is the point: `decode_all_terms` is now *literally*
`TermsCursor` collected —

```
let mut cursor = TermsCursor::open(data, entry)?;
while let Some(term) = cursor.next_term()? { terms.push(term.to_vec()); }
```

— so there is one decoder and not two, every existing corrupt-input guard and
error message is unchanged, and b4's whole negative-control suite (including
`blocktree`'s re-signed single-byte corruption sweep over the terms
dictionary) covers the cursor without a line of new fuzzing. The prefix copy
got *cheaper* on the way: the cursor truncates its own buffer to the prefix
length and extends, which is what `TermsDict.next()` does to its
`BytesRefBuilder`, where the materializer had to slice the previous term out
of the vector it was appending to.

Not spelled `next`, and not an `Iterator`: the returned slice borrows the
cursor's reused buffer, which is a lending iterator — the shape `Iterator`
cannot express and the shape that makes a full walk allocation-free. (Clippy's
`should_implement_trait` says the same thing.)

**Tested against real Lucene bytes.** `sorted_doc_values_fixtures.rs` now
walks the cursor over the same `.dvd` a real `IndexWriter` wrote and compares
against the *manifest*, so the cursor is checked against Java's output and not
against `decode_all_terms`. Checking it against `decode_all_terms` would prove
nothing — that function *is* the cursor collected, so such a test compares a
function to itself. The only cursor-specific unit test is therefore
`an_exhausted_terms_cursor_keeps_returning_none`, which is the one behaviour
the collected form cannot exercise and which §4's merge depends on.

---

## 4. `crates/lucene-search/src/ordinal_map.rs` — `build(TermsEnum[])`, at last

Java: `index/OrdinalMap.java`.

| Rust | Java | verdict |
|---|---|---|
| `TermCursor` (trait) | `TermsEnum`, restricted to `next()` | added |
| `OrdinalMap::build_streaming` | `OrdinalMap(owner, TermsEnum[], segmentMap, ratio)` | added — this is Java's actual constructor |
| `OrdinalMap::build` | `build(owner, SortedSetDocValues[], ratio)`'s materialized convenience | kept, now delegates |
| `MergeQueue` | `TermsEnumPriorityQueue` | added |
| `SliceCursor` | `TermsEnumIndex` over an array | added |
| `Cursor` (the old `BinaryHeap` element) | — | removed |
| the `top_term` scratch | `TermsEnumIndex.TermState topState` / `copyFrom` | identical role |
| `SegmentMap`/weights/`acceptableOverheadRatio` | `SegmentMap` | still not ported — unchanged, and still invisible in the output (module doc) |

### Finding 6 `[PERF → fixed]` the input was 5x the map

**Java**: `OrdinalMap.build` takes `TermsEnum[]`, primes a priority queue with
one term each, and never holds a dictionary. `topState.copyFrom(top)` is the
one owned copy, taken once per distinct term.

**We did**: took `&[Vec<T>]` — every segment's whole dictionary.

**Fixed**: `build_streaming(&mut [&mut dyn TermCursor])`. The one structural
difference from Java is forced by the borrow: a streaming cursor's term dies
at its next call, so the merge keeps one reused buffer per segment and the
queue orders *indices* into them — which is why `TermsEnumPriorityQueue`
became `MergeQueue`, a hand-rolled min-heap over `Vec<usize>` with the keys
outside it, rather than a `BinaryHeap`. Ties break to the lowest segment
index, which is what makes `first_segment` well-defined, unchanged from c12.
The segment ordinal is the push position rather than a `TermsEnum.ord()` call,
because a doc-values terms dictionary numbers its terms densely from zero in
enumeration order — taking it from the count removes an accessor
`TermsCursor` would otherwise have to carry.

`MergeQueue` carries Java's `updateTop` as well as `push`/`pop`: when the top
segment's cursor yields another term the key changes *in place*, so the drain
loop does one sift-down rather than a pop-then-push pair
(`OrdinalMap.java:315`). And `build_streaming` debug-asserts each cursor's
terms are strictly ascending as it advances, which is the check `build` does
over the lists up front and which a cursor gives no list to scan.

`OrdinalMap::build` keeps its signature for a caller that has the lists
anyway, and now runs **the same merge** over a slice-backed cursor. There is
one algorithm, not two, and
`build_streaming_agrees_with_build_over_the_same_terms` asserts that over four
shapes including an eight-segment one (deep enough that the heap compares both
children on the way down; a two- or three-segment case never reaches the right
child).

**The two halves are pinned together on real bytes.**
`facets_fixtures::global_counts` — the whole faceting read path — now opens
one `terms_dict::TermsCursor` per segment straight off the fixture's `.dvd`
and calls `build_streaming`, which is the only place the
`TermCursor for TermsCursor` adapter is exercised and the place the memory win
would actually land in-repo. Its assertions are against
`ordmap.seg.N.to_global`, the table real Lucene's own
`MultiDocValues.getSortedSetValues(...).mapping` wrote, so the streaming path
is pinned to **Java's** answer rather than to `OrdinalMap::build`'s; the two
entry points are additionally asserted equal on the same input.

**Measured**, one arm per process, 5 segments × 1 M 17-byte terms, 1.14 M
global ordinals, min of three runs each:

| arm | peak RSS over baseline | map | build | total (terms produced + built) |
|---|---|---|---|---|
| materialized | **318.2 MB** | 51.2 MB | 99–114 ms | **290 ms** |
| streaming | **51.4 MB** | 51.2 MB | 139–143 ms | **140 ms** |

**6.2x lower peak, and 2.1x faster end to end.** (Re-run after the Tier-2
review added `updateTop`, which took the `build` column from 154–158 ms to
139–143 ms.) The `build` column looks like a
regression and is not a comparison: the materialized arm's timer starts after
its 5 M terms have been produced into 5 M `Vec<u8>`s, and the streaming arm's
timer includes producing them. The total column is the honest one, and
producing-and-freeing 267 MB is most of what it removes. The map itself is
byte-identical between the arms, asserted where it can be: the example's two
arms run in separate processes and so compare nothing, but
`build_streaming_agrees_with_build_over_the_same_terms` asserts the whole map
over four shapes, and `facets_fixtures::global_counts` asserts it over the
real fixture *and* against Java's own `ordmap.seg.N.to_global` table.

### Finding 7 `[PERF → fixed]` `facets::resolve_labels` materialized the dictionary to walk it once

**We did**: `decode_all_terms` into a `Vec<Vec<u8>>`, then
`.zip(counts).enumerate()` — a strictly sequential ordinal-order walk that
held a second copy of every label beside the `FacetCount`s it was building.
c29's handoff named this caller alongside `OrdinalMap`.

**Fixed**: walks `TermsCursor` in ordinal order, with the "stop at the shorter
of the two" behaviour the `zip` had, now explicit and tested
(`more_counts_than_terms_stops_at_the_last_term`).

---

## Coverage

Every touched file, lines (the metric invariant #8 and `--fail-under-lines`
mean), from `cargo llvm-cov --workspace --summary-only`:

| file | lines |
|---|---|
| `lucene-codecs/src/postings_writer.rs` | 99.41% |
| `lucene-codecs/src/terms_dict.rs` | 99.03% |
| `lucene-index/src/indexing_chain.rs` | 98.12% |
| `lucene-index/src/index_writer.rs` | 98.32% |
| `lucene-index/src/merge.rs` | 98.60% |
| `lucene-search/src/ordinal_map.rs` | 99.03% |
| `lucene-search/src/facets.rs` | 99.49% |
| `lucene-index/src/check_index.rs` | 98.27% |

Workspace total 98.12% lines, unchanged.

## Verdicts

- `indexing_chain.rs` — **open, and now sized**: item 15 is a milestone
  (10 Java classes / 2 800 lines, three consumers, plus a `lucene-analysis`
  API change), with 36.8 MB + 36.6 MB named as the target and
  `examples/invert_memory.rs` as the instrument. Item 16 is closed here.
- `postings_writer.rs` — swept clean for this batch's concern; the flat
  payload run is the representation, and `permute_payload_run` is shared.
- `terms_dict.rs` — swept clean. Seeking (`seekExact`/`seekCeil`/`ord`)
  remains unported with no caller, unchanged from b4.
- `ordinal_map.rs` — swept clean. The one remaining recorded divergence is
  `Vec<i64>` rather than `PackedLongValues`, which is now the *larger* half of
  what is left rather than the smaller half of the peak.
- `facets.rs` — the label walk is streamed; the rest is untouched.
- `merge.rs`, `index_writer.rs`, `check_index.rs` — touched only where the
  payload representation crosses them.

## Handoffs

- **Item 15 (the block pool) is a milestone.** It needs planning as one, and
  the three-part shape above is the plan's skeleton: pools + `BytesRefHash`
  first, `FreqProxFields` second, the borrowed-token `Analyzer` API third
  (and that third part is independently worth measuring — one `String`
  allocation per token, on every field, is paid before this module is
  reached).
- **`c37-search-behaviours` has no row in `LEDGER.md`'s batch table.** Every
  other batch does. Not fixed here because its finding counts are its own to
  state, but it is the kind of record gap c34 existed to catch.
