# M1.6 sweep findings

One section per file swept, in the order of
[`docs/milestones/m1-6-lucene-sweep.md`](../milestones/m1-6-lucene-sweep.md)'s task
list. Lucene source read is the `releases/lucene/10.5.0` tag.

Every optimisation finding carries a measurement. Ratios are
`java_ns / rust_ns` on the same generated input via `scripts/bench-micro.sh`:
**above 1.0 means this port is faster than Lucene**.

---

## `crates/lucene-codecs/src/for_util.rs`

**Lucene counterpart:** `codecs/lucene104/ForUtil.java`, `PForUtil.java`,
`internal/vectorization/PostingDecodingUtil.java`,
`internal/vectorization/MemorySegmentPostingDecodingUtil.java` (java21).

**Measurement:** `scripts/bench-micro.sh --bench for_decode` — Rust
`benchmarks/rust-runner/src/micro.rs` against Java
`benchmarks/micro/java/org/apache/lucene/codecs/lucene104/ForUtilMicro.java`,
one decoded 256-value block per op, `bitsPerValue` 1..=31, both sides fed the
same xorshift-generated values and both round-trip-checked before timing.

### Baseline

| | median | mean |
|---|---|---|
| before any change | **0.75x** | 0.76x |

Rust was a third slower than Lucene in the innermost decode loop of the engine.
Worst cases were `bitsPerValue` 17-23 at 0.61-0.65x.

### P1 — parity: `bitsPerValue == 32` is not a Lucene-representable width

`ForUtil.decodeSlow` indexes `MASKS32`, declared `new int[32]`, so
`bitsPerValue == 32` throws `ArrayIndexOutOfBoundsException` in Lucene. This
port's `mask32` saturates instead and decodes 32 without complaint.

Found by the Java harness crashing on its first `bits32` case — not by reading,
which is worth noting, because the round-trip test on the Rust side passes at 32
and always would have.

**Read side: harmless** and deliberately kept — being able to decode a width
Lucene cannot produce costs nothing and the branch is tested.
**Write side: not reachable today.** `bits_required` can return 32, but only for
a value `>= 2^31`; doc deltas are bounded by `maxDoc <= 2^31-1` and freqs never
approach it. Filed rather than fixed: a guard would be dead code, and the real
protection is that `Verify*.java` reads everything this port writes.

### P2 — parity: doc-delta encoding choice picks FOR where Lucene picks the bit set

`Lucene104PostingsWriter` (line 444):

```java
} else if (numBitsNextBitsPerValue <= docRange) {
```

`postings_writer.rs::write_full_block`:

```rust
} else if num_bits_next_bits_per_value <= num_bit_set_longs * 64 {
```

`num_bit_set_longs * 64` is `ceil(docRange/64)*64`, which is `>= docRange`. So
for every block whose `docRange` falls in the band where
`docRange < numBitsNext <= ceil(docRange/64)*64`, this port writes packed FOR
deltas where Lucene writes the unary bit set.

Both encodings are legal and this port's reader accepts both, so nothing fails —
which is exactly why no test caught it. But Lucene's comment says the choice is
deliberate and which way it is meant to lean:

> we make the decision based on storage requirements, picking the bit set
> approach whenever it's more storage-efficient than the next number of bits per
> value (which effectively slightly biases towards the bit set approach)
>
> FOR makes `#nextDoc()` a bit faster while the bit set approach makes
> `#advance()` usually faster and `#intoBitSet()` much faster

So the divergence biases the wrong way for exactly the operation M1 and M1.5 were
about. **Status: open, to fix in Stage B** alongside the rest of the postings
sweep, because it wants a writer test that pins the chosen encoding per block
rather than only that the bytes round-trip.

### O1 — optimisation: `split_ints` used the pre-vectorization loop nest

`split_ints` iterated `for i { for j }`. Lucene iterates `for j { for i }`, and
its own comment says why: *"Process each shift level across all elements (better
for vectorization)"*. The transposed nest gives the inner loop a variable trip
count and a stride-`count` write pattern, which is what stops a vectoriser.

Fixed: shift level outer, elements inner, iterating over sub-slices with `zip`
so the bounds checks hoist out of the loop rather than repeating per element.

| | median | mean |
|---|---|---|
| before | 0.75x | 0.76x |
| after O1 | **1.47x** | 1.64x |

A ~2x improvement on the kernel from reordering two loops. `bitsPerValue` 11-15
moved furthest (0.73-0.80x to 2.48-3.23x): those widths have the most shift
levels per word, so they had the most to gain from a vectorisable inner loop.

### O2 — optimisation: scratch buffer allocated per call, not per decoder

Lucene's `ForUtil` holds `private final int[] tmp = new int[BLOCK_SIZE]`,
allocated once and reused for every block it ever decodes. This port declared
that buffer inside `for_decode`, so every 256-value block paid a 1 KiB
zero-fill, and `decode_slow` allocated a second one on top for
`bitsPerValue > 16`. `encode_generic` went further and did a heap `vec![]` per
call.

Fixed by giving the port the same shape Lucene has: a `ForUtil` struct owning
the scratch, with `decode`/`pfor_decode` methods. The free `for_decode`/
`pfor_decode` functions stay for one-shot callers and construct one per call, so
no caller broke; `encode_generic`'s `vec![]` became a stack array.

| | median | mean |
|---|---|---|
| after O1 | 1.47x | 1.64x |
| after O2 | **1.62x** | 1.75x |

Small widths gained most (`bits01` 22.4 -> 14.8 ns, `bits04` 34.5 -> 26.1 ns):
the fixed 1 KiB zero-fill is a larger share of a cheap decode than an expensive
one.

Note the harnesses were updated in the same step to hold one `ForUtil` across
iterations, matching the Java harness's single `new ForUtil()` per case. Before
that they charged the Rust side a zero-fill Lucene never pays — the comparison
was measuring a workload the engine does not run.

### O3 — optimisation: no bulk word read

`split_ints` read its `count` words with `count` separate `read_u32_le()` calls,
each with its own bounds check and `Result`, where Lucene issues one
`in.readInts(c, cIndex, count)`. `decode8`/`decode16`, which are *nothing but*
that read, showed the cost directly: `bits08` at 0.48x was the one case that got
*worse* across O1/O2 (19.9 -> 45.5 ns) while every other case improved. It does
not call `split_ints` at all, so neither change touched it — the regression was
codegen fallout from `split_ints` becoming vectorisable and inlining differently
across the 30-arm dispatch, which is only possible because `decode8` was a
64-iteration loop rather than a copy.

Fixed by adding `DataInput::read_u32s_le`, the port's `readInts`: the default is
the naive loop exactly as Lucene's default is, and `SliceInput` overrides it with
one bounds check for the whole run followed by a `chunks_exact(4)` copy that LLVM
lowers to a bulk move on a little-endian target. `split_ints`, `decode8` and
`decode16` all use it.

| | median | mean |
|---|---|---|
| after O2 | 1.62x | 1.75x |
| after O3 | **2.26x** | 2.24x |

`bits08` 45.5 -> 14.2 ns (0.48x -> 1.54x), `bits16` 19.5 -> 14.6 ns
(0.89x -> 1.19x). **Every one of the 31 widths is now above 1.0x**, which was not
true at any earlier point in the sweep.

### Where `for_util.rs` ended up

| stage | median | mean | widths below Lucene |
|---|---|---|---|
| baseline | 0.75x | 0.76x | 29 of 31 |
| O1 loop nest order | 1.47x | 1.64x | 5 |
| O2 reusable scratch | 1.62x | 1.75x | 3 |
| O3 bulk word read | **2.26x** | 2.24x | **0** |

3.0x faster than where this file started, and 2.26x faster than Lucene's
Panama-vectorised decode, from three changes none of which is SIMD.

The weakest remaining widths are 17-19 at 1.04-1.12x — the start of the
`decode_slow` range, whose tail loop stitches values across word boundaries one
at a time and is inherently serial.

### O4 — optimisation: no SIMD (deferred, with reason)

Lucene ships `MemorySegmentPostingDecodingUtil`, which loads `IntVector`s
straight out of the mapped segment and does the shift/mask lanewise, selected at
runtime by `VectorizationProvider`. This port has no explicit SIMD and no
dispatch layer to hang one on; it relies on LLVM auto-vectorising the scalar
loops.

**Deferred, and this is a measured decision rather than a punt.** After O1-O3 the
scalar Rust kernel beats Lucene's vectorised one on all 31 widths by a median of
2.26x. Adding explicit SIMD would mean relaxing `#![forbid(unsafe_code)]` on
`lucene-codecs` — `core::arch` intrinsics are `unsafe` — plus a runtime feature
dispatch and a second implementation of the kernel to keep in agreement with the
first, forever. That is a real and permanent cost against a kernel that is
already ahead.

Revisit if a future finding shows postings decode back on the critical path. The
honest reading of these numbers is that the original 0.75x was never a SIMD
problem: it was a loop written in the order that prevents vectorisation, a buffer
allocated in the wrong scope, and a missing bulk primitive.

---

## `crates/lucene-codecs/src/postings.rs`

**Lucene counterpart:** `codecs/lucene104/Lucene104PostingsReader.java`
(`BlockPostingsEnum`), `Lucene104PostingsWriter.java`.

**Measurement:** `scripts/bench-micro.sh --bench postings_iter` — walk every
segment's posting list for a term with `nextDoc()` until exhausted, reported as
ns per document. Rust drives `LazyDocsCursor`; Java drives
`Lucene104PostingsReader`'s `BlockPostingsEnum` through the public
`TermsEnum.postings()` API, because unlike `ForUtil` nothing is package-private
in the way and that is how a real query reads postings. Both sides ask for
`PostingsEnum.NONE` (docs only) and both read the same `benchmarks/.corpus/merged`
directory, so they walk identical on-disk bytes. Four terms spanning three orders
of magnitude of list length, since a single term measures one block-encoding
shape.

### O5 — optimisation: `next_doc` binary-searched for an offset that is always 1

**This is the M1 gap.**

| term | before | after | Lucene | before | after |
|---|---|---|---|---|---|
| `t0`  | 14.48 ns | 1.86 ns | 2.90 ns | 0.20x | **1.56x** |
| `t1`  | 14.68 ns | 2.16 ns | 3.00 ns | 0.20x | 1.39x |
| `tz`  | 14.84 ns | 2.39 ns | 3.47 ns | 0.23x | 1.45x |
| `t2s` | 14.17 ns | 1.76 ns | 1.84 ns | 0.11x | 1.05x |
| median | | | | **0.20x** | **1.42x** |

`LazyDocsCursor::next_doc` was implemented as `advance(doc_id + 1)`. That is
correct, and it is how the contract reads, but `advance` begins with a
`partition_point` over the remaining entries of the decoded block — a binary
search over up to 256 elements, eight unpredictable branches — to compute an
offset that in the `nextDoc` case is always exactly 1.

Lucene's is three lines and no search:

```java
public int nextDoc() throws IOException {
  if (doc == level0LastDocID) { moveToNextLevel0Block(); }
  return this.doc = docBuffer[docBufferUpto++];
}
```

Fixed by giving `next_doc` the same shape: take the next slot when the decoded
block still has one, fall back to `advance` only at a block boundary and before
the first call, where the state machine actually lives. The invariant this rests
on — `block_docs[block_pos] == doc_id` whenever `block_pos < block_len` — is
established by every path in `advance` that sets `doc_id` to a real document, and
is now stated at the fast path.

**7.8x on the operation M1 exists to measure**, and 0.20x -> 1.42x against Lucene.

Worth recording how long this hid. M1 profiled the end-to-end query benchmark and
got a flat profile — largest item 14.78% — and concluded no single mechanism was
left. M1.5 rebuilt iteration around `LazyDocsCursor` specifically to stop
materializing posting lists. Neither found this, because a profile attributes the
cost to `advance`, which is a function that legitimately needs to be there and
legitimately does a search. Only comparing *the same operation* against Lucene's
number for it makes 14.5 ns against 2.9 ns visible as a defect rather than as
what iteration costs.

It is also why the `for_util.rs` work above, real as it is, moved the end-to-end
benchmark far less than its 3x suggested: block decode was never where the time
went.

### O6 — optimisation: a full-block decode allocated ~8 KiB of scratch per block

`decode_full_block_body` declared `docs`, `freqs`, `doc_deltas` and `freq_words`
as locals and returned the first two **by value**, so every 256-document block
paid for roughly 8 KiB of zeroed and copied stack to produce 2 KiB of output,
plus a `vec![]` on the dense bit-set path and (after the `ForUtil` work above) two
freshly zeroed `ForUtil` scratch buffers.

Lucene's `BlockPostingsEnum` holds every one of these as an instance field for the
life of the enumeration. Introduced `BlockScratch` and gave it the same lifetime:
one per `LazyDocsCursor`, one per eager `read_postings` call rather than one per
block. The dense path's bit set became a fixed `[u64; 128]` array — `-bitsPerValue`
is read from an `i8`, so `numLongs` can never exceed 128 however corrupt the file
is, which is also Lucene's own bound (`assert numBitSetLongs <= BLOCK_SIZE / 2`).

Folded into the same measurement as O5 and not separately attributable; the
`next_doc` fix dominates it by an order of magnitude.

### P2 — parity: doc-delta encoding choice (fixed)

Recorded under `for_util.rs` above as found. `postings_writer.rs::write_full_block`
compared `numBitsNextBitsPerValue` against `num_bit_set_longs * 64` where Lucene
compares against `docRange` itself; the former is the latter rounded up to whole
words, so every block landing strictly between the two got packed FOR deltas where
Lucene writes the unary bit set — the encoding Lucene's own comment says is chosen
to make `advance()` and `intoBitSet()` faster.

Fixed to Lucene's condition. The new test
`full_block_encoding_choice_matches_lucene_in_the_disputed_band` constructs a block
in the band (208 deltas of 3 and 48 of 2: `docRange == 720`,
`numBitsNext == 768`, `bits2words(720) * 64 == 768`) and asserts the **chosen
token**, not just that the bytes round-trip. Negative control run: with the old
condition restored it fails with `left: 2, right: -12`, exactly the predicted
divergence.

---

## `crates/lucene-search/src/lib.rs` (term MAXSCORE loop), `collector.rs`, `field_norms.rs`

**Lucene counterpart:** `search/ImpactsDISI.java`, `search/MaxScoreCache.java`,
`search/TopScoreDocCollector.java`, `search/similarities/BM25Similarity.java`.

**Measurement:** `scripts/bench-compare.sh` on `q01` (`body:t0`, the most
frequent term, top-50), plus `perf record` profiles between each change. The
component benchmarks above could not have found any of these: they are all in the
search layer, above the codec.

### Why look here at all

After the postings work the end-to-end benchmark had barely moved, which was
itself the finding. A profile of `q01` put roughly half the runtime above the
codec:

| symbol | share |
|---|---|
| `search_term_query_scored_maxscore_with_stats` (+ its closure) | 26.2% |
| `decode_full_block_body` | 10.0% |
| `decode_impacts` | 9.8% |
| `TopDocsCollector::collect` | 7.8% |
| `ForUtil::decode` | 5.1% |
| `norms::norm_value` + `read_value_at_ordinal` | 7.6% |

### O7 — impacts were decoded eagerly, per block, into a fresh `Vec`

`read_full_block_header` decoded the impacts byte run into a freshly allocated
`Vec<Impact>` on **every** block header it read — including every block the
cursor went on to skip without ever asking for a bound. Lucene keeps the run
undecoded as a `BytesRef` (`level0SerializedImpacts`) and calls `readImpacts`
into a reusable `FreqAndNormBuffer` only when `getImpacts()` asks.

Fixed: `FullBlockHeader` carries `impact_bytes: &'a [u8]`, a zero-copy borrow of
the mapped `.doc` file (better than Lucene, which copies into its `BytesRef`),
and the cursor decodes into its own reused buffer at the one point a block is
actually loaded. `decode_impacts_into` is the reusable-buffer form;
`decode_impacts` remains for callers that want an owned list.

`q01`: **525 -> 646 qps** (+23%).

### O8 — the impact bound was evaluated once per document, not once per block

The scoring loop asked `collector.min_competitive_score()`, fetched the block
impacts and consulted a cached bound **for every document**. Lucene's
`ImpactsDISI.advanceTarget` opens with:

```java
if (target <= upTo) {
  // we are still in the current block, which is considered competitive
  // according to impacts, no skipping
  return target;
}
```

so the bound is evaluated per *block*, not per document — 256x fewer times.

The other half of that mechanism is not optional, and a test caught its absence.
`ImpactsDISI.setMinCompetitiveScore` sets `upTo = -1` whenever the threshold
actually rises, so a block judged competitive against an old threshold is
re-judged against the new one. Implementing only the `upTo` check made the loop
stop skipping blocks entirely on a two-block fixture — the *results* were
unchanged, only the work done differed, and it was
`maxscore_lazy_path_matches_eager_path_on_real_fixture_and_actually_skips_blocks`'s
skip counter that failed rather than any assertion about output. That test exists
because a skip branch can go dead without any result changing; this is the second
time it has earned its keep.

`q01`: **646 -> 708 qps**.

### O9 — `TopDocsCollector::collect` had no fast reject

`TopScoreDocCollector.collect` opens with one float comparison against the
queue's worst hit and returns. This port built a `ScoreDoc`, then called
`rank_order` twice through the general insert path, for every document.

Added the same first line, keeping the general path (and its `total_cmp` NaN
ordering) for anything that survives it. In the profile `collect` went 9.5% ->
5.4%.

Note what is *not* fixed: `TopDocsCollector` is still a sorted `Vec` with an
`O(n)` insert where Lucene uses a binary heap. That is a documented decision in
the type's own doc comment and it does not show in this profile, because with the
fast reject in front of it an insert is rare. Left alone.

### O10 — norm lookup went through the general doc-values decoder per document

`FieldNorms::norm_inverse` called `norms::norm_value`, which re-tests denseness
and norm width, then constructs a `SliceInput` and seeks it — to read one byte at
a known offset. That is 9.5% of the profile to index an array.

`Lucene90NormsConsumer` writes a flat one-byte-per-document array for any
ordinary analyzed field. Resolve that slice once when the `FieldNorms` is built,
and scoring a document becomes `table[bytes[doc]]`, which is what `BM25Scorer`
reading `cache[norm]` is. Sparse fields, wider norms, constant-valued fields and
empty fields decline the fast path and take the general one.

Three tests pin it, because two implementations of one lookup is exactly the
shape that silently diverges: agreement with the general path for all 256
possible norm bytes, out-of-range documents still erroring rather than reading
past the array, and the declined shapes actually declining.

### Where the term path ended up

| query | before | after |
|---|---|---|
| `q01` `body:t0` | 0.53x | **0.82x** |
| `q02` `body:t1` | 0.37x | 0.53x |
| `q03` `body:tz` | 0.29x | 0.35x |
| `q04` `body:t2s` | 0.39x | 0.58x |
| `q05` `body:t1z4` | 0.48x | 0.72x |
| `q18` `title:t0` | 0.34x | 0.44x |
| `q19` `keyword:t0` | 4.53x | 4.24x |

Recall mismatches: **0**, across all 20 queries, both before and after.

**Still open.** The boolean and phrase queries (q06-q17) barely moved: they run
through `try_conjunction_lazy`/`try_disjunction_lazy` and the phrase scorer, none
of which have had the `ImpactsDISI` treatment, and the phrase queries at 0.04-0.07x
are their own problem in the positions reader. That is the next stage of the
sweep, not a conclusion.

---

## `crates/lucene-search/src/lib.rs` (boolean lazy paths)

**Lucene counterpart:** `search/BM25Similarity.java` (`BM25Scorer`),
`search/ImpactsDISI.java`, `search/BlockMaxConjunctionScorer.java`,
`search/WANDScorer.java`.

**Measurement:** `scripts/bench-compare.sh`, plus `perf record -s srcline` on
`q11` (`or body:tz body:t2s`, the worst non-phrase query at 0.08x).

### O11 — `idf` was a `ln()` recomputed per document, per clause

The disjunction and conjunction scoring loops both called
`similarity::score_with_params(doc_freq, doc_count, ...)`, whose first act is
`idf(doc_freq, doc_count)` — a `ln()`. `libm`'s `log` was **over 15% of a
two-clause disjunction's entire profile**.

Lucene computes idf once per term in `BM25Similarity.scorer` and carries it as
the scorer's `weight`. Both `Leg` structs now do the same, and the score becomes
`leg.weight * tf_norm(...)` — the same factors in the same order, so the result
is bit-identical to what it produced, minus one `ln()` per document per clause.

| query | before | after |
|---|---|---|
| `q11` `or tz t2s` | 44.4 qps | 73.4 qps |
| `q12` `or t0 t1 t2 t3` | 5.8 qps | 12.1 qps |
| `q06` `and t0 t1` | 79.3 qps | 92.0 qps |

### O12 — `field_length` also went through the general doc-values decoder

Same finding as O10 and the same fix, for the accessor the boolean scorers use.
They sum per-clause contributions, so they keep the multiply form to stay
bit-identical; that needs the decoded length rather than the reciprocal, so
`FieldNorms` gained a `norm_length` table alongside `norm_inverse` and the same
dense one-byte fast path in front of it.

### O13 — the disjunction re-derived its span and bound per document

Same `ImpactsDISI.upTo` treatment as the term path (O8): a pass over every leg
to recompute `up_to`, then a cache probe, ran once per document to reach a
decision that only changes when a leg crosses a block boundary or the threshold
rises. Those two lines were 20% of `q11`'s profile.

`q11` 73.4 -> 83.2 qps, `q12` 12.1 -> 13.4 qps.

### O14 — the same treatment on the conjunction: **tried, measured slower, reverted**

Applying the identical restructure to `try_conjunction_lazy` cost 9%:

| query | span-keyed cache | `upTo`-keyed | |
|---|---|---|---|
| `q06` `and t0 t1` | 91.2 qps | 83.0 qps | -9% |
| `q07` `and t0 tz` | 167.4 qps | 144.7 qps | -14% |
| `q09` `and t0 t1 t2` | 24.1 qps | 22.8 qps | -5% |

The reason is specific to the shape: a leapfrog's `candidate` regularly
overshoots `up_to` — the skip is guarded on `up_to >= candidate` precisely
because it can — so keying the "already decided" marker on the *document* rather
than on the *span* invalidates it almost every iteration and loses the cache
entirely. The span-keyed cache already there does the same job for this shape.

Reverted, and the reason left in the code so the next reader does not re-derive
it. This is the third measured revert in this project's performance work
(header-only block skipping, WAND partitioning, this), and all three looked
obviously right beforehand.

### Where the boolean paths ended up

| query | before sweep | after |
|---|---|---|
| `q06` `and t0 t1` | 0.22x | **0.31x** |
| `q07` `and t0 tz` | 0.11x | 0.15x |
| `q08` `and tz t2s` | 0.17x | 0.18x |
| `q09` `and t0 t1 t2` | 0.14x | 0.22x |
| `q10` `or t0 t1` | 0.27x | 0.37x |
| `q11` `or tz t2s` | 0.08x | 0.16x |
| `q12` `or t0 t1 t2 t3` | 0.11x | 0.26x |
| `q20` `and title t0 t1` | 0.15x | 0.17x |

Recall mismatches: **0**.

### Still open, in priority order

1. **Phrase queries `q16`/`q17` at 0.03-0.04x.** Untouched by this sweep and now
   by far the worst thing in the benchmark — 25x slower than Java, where nothing
   else is worse than 6x. The positions reader is Stage B2 and has not been read
   against `Lucene104PostingsReader`'s `.pos`/`.pay` path at all yet.
2. **`tf_norm` is 14% of a disjunction's profile.** It computes
   `freq / (freq + k1*(1 - b + b*len/avgdl))` — two divisions. Lucene's
   `BM25Scorer.doScore` is `weight - weight / (1 + freq * normInverse)`: one
   division, with `normInverse` read from the table this port already builds.
   The term path already uses that form. Switching the boolean paths would move
   them *closer* to Lucene's arithmetic, not further, but it is not bit-identical
   to what they produce today, so it needs its expected values re-derived against
   `IndexSearcher.explain()` rather than against this port's own previous
   output. Filed rather than rushed.
3. **WAND essential/non-essential clause partitioning.** The disjunction's own
   comment already says this is what it does not do. It is the remaining
   structural difference from `WANDScorer` and is a milestone-sized piece of
   work, not a sweep finding.

---

## Phrase queries: `search_phrase_query_scored`, `term_doc_positions`

**Lucene counterpart:** `search/ExactPhraseMatcher.java`, `search/PhraseScorer.java`,
`codecs/lucene104/Lucene104PostingsReader.java` (the `.pos` reader).

### P3 — parity: phrase clauses scored from per-segment idf (fixed)

The 15-segment corpus was re-measured after the boolean work and came back with
**2** recall mismatches out of 20 — down from 13, and both of them the phrase
queries. Every other query agreed with Java exactly.

Same bug as the one at the top of this milestone, for a clause type the fix
missed. A phrase's idf is the sum of its constituent terms' idfs, and
`search_phrase_query_scored` was computing each from `field_terms.doc_count` and
that segment's own `docFreq`. `global_boolean_stats`'s walk only collected
`Clause::Term`, so no phrase term ever got a reader-wide entry.

Fixed on both sides: the walk now descends into `Clause::Phrase`,
`Clause::DisjunctionMax` and `Clause::Boost` as well as nested booleans, and
`search_phrase_query_scored_with_stats` / `term_doc_scores` take the map. That
last one closed a second gap in passing: a boolean query whose shape does not fit
a lazy path falls through to `clause_scores` -> `term_doc_scores`, which was
*also* still scoring terms per segment. The benchmark's and/or queries all fit a
lazy path, so nothing had caught it.

**Segmented corpus recall mismatches: 2 -> 0.** Both corpus variants now agree
with Java on every hit set, which is the first time that has been true.

Still per-segment, and listed here rather than left implicit: `Clause::Wildcard`,
`Prefix`, `Fuzzy` and `Regexp` expand to terms this port resolves elsewhere and
do not get reader-wide statistics yet. No benchmark query exercises them.

### O15 — phrase matching materializes every position of every document (open)

Phrase queries are now by far the worst thing in the benchmark: 0.03-0.04x on
the merged corpus, where nothing else is below 0.15x. A profile says why —
**about 50% of the query is in `malloc`/`free`/`memcpy`**:

| symbol | share |
|---|---|
| `libc` (allocator and `memcpy`) | ~52% |
| `term_doc_positions` | 11.9% |
| `clause_scores` | 4.3% |
| `resolve_clause_docs` | 3.6% |
| `Vec<Vec<Position>>::push` | 2.7% |

`term_doc_positions` builds a `Vec<Vec<i32>>`: one heap allocation per matching
document, for every document containing the term, before any phrase matching
starts. On `body:t0` that is roughly five million allocations per query.

This is exactly the defect M1 diagnosed and M1.5 fixed for the *doc* stream —
"stop materializing posting lists" — never done for the *position* stream.
Lucene's `ExactPhraseMatcher` walks `PostingsEnum.nextPosition()` lazily and
allocates nothing per document.

**Fixed, in the cheaper half.** Full laziness — a `LazyPositionsCursor`
mirroring `LazyDocsCursor`, so positions are never decoded for a document the
phrase cannot match — is still M1.5-sized and still open. But the *allocation*
half needed neither.

`read_positions`'s decode already builds a flat `pos_deltas: Vec<i32>`; only its
final assembly loop chopped that into a `Vec` per document. Split the wire-format
half out as `decode_position_streams`, and add `read_positions_flat`, which
assembles the same data as one positions array plus per-document start offsets:
**two allocations regardless of how many documents match**, and no offset or
payload fields the phrase matcher never reads. `FieldTerms::positions_flat` and
`term_doc_positions` follow, the latter now returning
`(docs, positions, spans)`.

`q16` `phrase t0 t1`: **0.1 -> 0.5 qps**.

### O16 — the phrase query was executed twice

With the allocator out of the profile, two callers stood out:
`resolve_clause_docs` at 14.2% and `clause_scores` at 17.6%. They are the two
halves of `search_boolean_query_scored`'s general path — find the matching
documents, then score them — and for a query that is a *single* scoring clause
they do the same work twice, including a full second pass of position decode.
The multi-segment layer reaches phrase queries exactly that way:
`BooleanQuery { must: vec![Clause::Phrase(..)] }`.

A single `Term` or `Phrase` `must` clause with nothing to filter against needs no
matching pass at all: the clause's score map *is* its matched set. Restricted to
those two clause kinds deliberately — a nested `Boolean` can match documents its
scoring sub-clauses never mention, and the wildcard family expands to terms
elsewhere, so both keep the two-pass path — and to `minimum_should_match == 0`
rather than `<= 1`, because with no `should` clauses a minimum of 1 means nothing
matches and the fast path would wrongly return the `must` clause's documents.

`q16`: **0.5 -> 0.9 qps**.

### Where the phrase path ended up

| query | before | after | |
|---|---|---|---|
| `q16` `phrase t0 t1` | 0.1 qps (0.04x) | **0.9 qps (0.31x)** | 9x |
| `q17` `phrase t1 t2` | 0.3 qps (0.07x) | **1.7 qps (0.38x)** | 5.7x |

Phrase queries are no longer the outlier: at 0.31-0.38x they now sit above the
boolean queries rather than an order of magnitude below everything.

What remains in the profile is `term_doc_positions` itself — decoding every
position of every document containing each term, before any of them is known to
be a phrase candidate. That is the part that needs the lazy cursor, and it is
still open.

---

## `crates/lucene-codecs/src/direct_reader.rs`

**Lucene counterpart:** `util/packed/DirectReader.java` (its fourteen
`DirectPackedReaderNN` classes), `util/packed/DirectWriter.java`.

**Measurement:** `scripts/bench-micro.sh --bench direct_reader` — a fixed odd
stride through a packed array of 2^17 values, one `get(index)` per op, every
width `DirectWriter` supports. Strided rather than sequential deliberately: this
primitive serves random per-document lookups, and a sequential sweep would
measure the hardware prefetcher.

### O17 — `get` read one byte at a time

`direct_reader::get` assembled its word with `for (i, &b) in bytes.iter()`: one
load, one shift and one OR per byte. Lucene's `DirectPackedReaderNN` classes each
issue a single `readInt`/`readLong`.

The shape of the measurement is the finding — our cost rose linearly with the
width while Lucene's stayed flat:

| bits | rust before | java | | rust after | |
|---|---|---|---|---|---|
| 1 | 1.96 ns | 0.68 ns | 0.35x | 1.22 ns | 0.56x |
| 8 | 1.98 ns | 2.21 ns | 1.12x | 1.24 ns | 1.79x |
| 32 | 3.18 ns | 2.33 ns | 0.74x | 1.41 ns | 1.65x |
| 64 | **5.25 ns** | 2.34 ns | 0.45x | **1.42 ns** | 1.64x |
| median | | | **0.82x** | | **1.82x** |

Fixed with a single 8-byte little-endian load whenever eight bytes are in range,
falling back to the byte loop at the tail. Eight bytes always suffice for the
supported widths: the non-byte-aligned ones stop at 28 bits (five bytes at
worst) and the byte-aligned ones never carry a shift. This is also what
`DirectWriter`'s output padding is *for* — `padding_bytes_needed` in this same
file is already this port's copy of that rule, and `Lucene90DocValuesConsumer`
writes the padding for exactly this reason.

Now flat at ~1.23 ns across all fourteen widths, 3.7x faster than before at 64
bits and 1.82x Lucene overall. `bits01`/`bits02` remain below 1.0x; Lucene's
0.68 ns for a one-bit read is faster than a dependent load can be, so that
column is measuring C2 folding the specialized reader away rather than measuring
a read.

This is not on the measured query path — doc values, sorting and facets use it,
and none of M1's 20 queries do. It was found by reading, and it is recorded here
with that caveat rather than being claimed as an end-to-end improvement.

---

## `crates/lucene-store/src/directory.rs`

**Lucene counterpart:** `store/MMapDirectory.java`, `store/MemorySegmentIndexInput.java`.

### Checked, no gap: `madvise`

Lucene's `MMapDirectory` can call `madvise()` with per-file read advice, and
codecs pass hints (`Lucene90CompressingStoredFieldsReader` asks for
`DataAccessHint.RANDOM`, merges and flushes get `SEQUENTIAL`). This port calls
none, which looked like a gap.

It is not one at the default. `MMapDirectory.readAdvice` initialises to
`(filename, context) -> Optional.empty()` — no advice at all unless a caller
installs `ADVISE_BY_CONTEXT` or its own function. The hints the codecs pass are
inert until someone does. So the out-of-the-box behaviour matches, and the
honest finding is that per-file advice is a *feature* this port does not offer,
not a performance defect it is suffering from.

Recorded rather than acted on, and deliberately not guessed at: `MADV_RANDOM`
disables kernel readahead, so applying it blind could easily cost more than it
saves. Verifying it needs a cold-page-cache benchmark, which the current harness
is not — every measurement in this milestone runs against a warm 1.6 GB index.

Still open in this file: `IndexInput.prefetch` (Lucene 10's explicit
`madvise(WILLNEED)`, used to overlap IO with decode) has no equivalent here.
Same caveat — it cannot be measured warm.

---

## `crates/lucene-codecs/src/blocktree.rs`

**Lucene counterpart:** `codecs/lucene90/blocktree/Lucene90BlockTreeTermsReader.java`,
`SegmentTermsEnum.java`, `SegmentTermsEnumFrame.java`.

**Measurement:** `scripts/bench-micro.sh --bench reader_open --index <dir>` —
open a reader over the 15-segment, 5M-document corpus and count its segments,
once per op. Rust runs `DirectoryReader::open` plus `open_segments`; Java runs
`DirectoryReader.open`.

### A1 — architecture: the whole term dictionary is materialized at open

| | ns per open | |
|---|---|---|
| Java `DirectoryReader.open` | 4,153,560 | 4.2 ms |
| this port | 559,656,637 | **560 ms** |
| ratio | | **0.01x — 135x slower** |

`blocktree::FieldTerms` holds
`entries: Vec<(Vec<u8>, TermStats, TermMetadata)>` — **every term of the
field**, each with its own heap allocation — built eagerly when the segment is
opened. `seek_exact` is then a binary search over that vector.

Lucene holds none of it. `SegmentTermsEnum` walks the `.tip` FST to locate the
on-disk block that could contain a term, loads that one block's suffix bytes
into a reusable frame, scans them in place, and decodes term metadata only for
the term actually being sought. Its memory is O(1) in the vocabulary and its
open cost is O(number of fields), not O(number of terms).

**This is the largest architectural divergence left in the read path**, and no
query benchmark could have found it: the reader is opened once, outside the
timed region, in both `bench-compare.sh` and every fixture test. It took
measuring an operation that is not a query.

It is not an academic difference. A search engine reopens readers on every
refresh — that is what a refresh *is* — so 560 ms of per-reopen cost is
disqualifying for M2 and M5 regardless of how fast queries then run. The memory
side is worse in a way this benchmark does not show at all: one `Vec<u8>` per
term per field per segment, live for as long as the reader is.

**Filed, not fixed.** Replacing it means implementing real block-tree
navigation — FST arc walking to a block, frame-based suffix scanning, lazy
metadata decode — which is a milestone in its own right and touches the largest
file in the codec crate. The sweep's contribution is the number: 135x, on an
operation nobody had measured.

### D2 — `fst.rs` is complete, fixture-verified, and unused by the read path

Entailed by the above rather than a separate finding, but worth stating: this
port has a full FST implementation, byte-compatible and differentially verified
against Java (`write_fst_fixture` / `VerifyFst`), which **the term dictionary
never calls**. The eager traversal replaced it. Today only `suggest.rs` uses it.

Real block-tree navigation would route through it, so the work above is not a
rewrite from nothing — the piece it needs already exists and is proven.

This also reframes two things recorded earlier in this document. `FieldTerms`
having every term in memory is why `seek_exact` never appears in any query
profile — the work was already done, before the clock started. And it is why
`positions_flat`/`postings` can hand back owned `Vec`s so freely: the type
already owns everything.

---

## `crates/lucene-codecs/src/indexed_disi.rs`, `norms.rs`

**Lucene counterpart:** `codecs/lucene90/IndexedDISI.java`,
`Lucene90NormsProducer.java`, `Lucene90DocValuesProducer.java`.

**Measurement:** `indexed_disi/sparse_lookup` in
`crates/lucene-codecs/benches/hot_paths.rs`. Rust-only, deliberately: the point
is the shape of the curve on this side, not a ratio.

### A2 — architecture: a sparse lookup decodes the whole doc-id list, every time

| documents with the field | one lookup |
|---|---|
| 1,000 | 874 ns |
| 10,000 | 31.2 us |
| 100,000 | **324 us** |

Linear in the field's cardinality, which is the defect. `indexed_disi::
decode_doc_ids` decodes the **entire** `IndexedDISI` region into a fresh
`Vec<i32>` and the caller then binary-searches it — and every sparse lookup in
the port calls it: `norms::norm_value`, and three sites in `doc_values.rs`.

Lucene's `IndexedDISI` is a forward-only iterator with a jump table;
`advance(target)` is roughly constant time and allocates nothing.

So sorting or faceting on a sparse field is quadratic in this port. At 100,000
present documents, one pass over them costs on the order of 30 seconds of pure
DISI decoding. No test or benchmark here exercises that shape, which is why it
had never been seen: M1's corpus fields are all dense.

### O18 — fixed where there is an owner: `FieldNorms`

`FieldNorms` is constructed once per field per segment per search and then asked
for a norm per document, so it is exactly the place to decode once. It now holds
`sparse_doc_ids: Option<Vec<i32>>`, decoded in the constructor, and
`norm_inverse`/`field_length` resolve a sparse document through
`rank_of` + `norms::read_value_at_ordinal` instead of re-decoding. A lookup goes
from O(cardinality) to a binary search.

Tested the same way as the dense fast path: the sparse branch is asserted to
agree with `norms::norm_value` for every document, present and absent alike, and
the values are additionally checked against the bytes actually written so the
test cannot pass by both paths being wrong identically.

**This is a fix at the caller, not at the defect.** The three `doc_values.rs`
sites are free functions with no natural owner and are unchanged; sorting and
faceting on a sparse field are still quadratic.

### O20 — the `IndexedDISI` cursor (fixed)

`indexed_disi::DisiCursor` is now the forward-only reader this finding asked
for: it walks at most one block header per 65,536 documents and scans one block,
allocating nothing. All four sparse sites use it -- `norms::norm_value`, and
`doc_values`' `numeric_value`, `sorted_ord`/`binary_value` path and
`sorted_numeric_values`.

| documents with the field | decode-all | cursor | |
|---|---|---|---|
| 1,000 | 843 ns | 509 ns | 1.7x |
| 10,000 | 31.4 us | 258 ns | **122x** |
| 100,000 | 326 us | 178 ns | **1,832x** |

Flat where the old path was linear, which is the whole claim; the benchmark
measures both shapes side by side so the contrast is visible rather than
asserted.

Forward-only is **enforced**, not merely documented. Going backwards happened to
work within a block -- a SPARSE block rescans from its start, a DENSE one indexes
by bit position -- and failed across one. That is the worst kind of contract:
correct in testing, wrong on the data that spans two blocks. A test that went
backwards deliberately is what surfaced it.

Correctness is pinned by asserting the cursor agrees with `decode_doc_ids` +
`rank_of` for **every** document in range, not at sampled points, across all
three block encodings (SPARSE, DENSE, ALL) and across a block that carries no
values at all -- the case a "step to the next block" loop gets wrong. The
interesting answers are the `None`s, which sampling would miss.

`FieldNorms` keeps its decode-once `Vec`: it is built per field per segment and
then asked for arbitrary documents, so random access matters more there than
allocation, and the cursor's forward-only contract would not serve it.

---

## `crates/lucene-codecs/src/doc_values.rs`

**Lucene counterpart:** `codecs/lucene90/Lucene90DocValuesProducer.java`
(`VaryingBPVReader`, `DenseNumericDocValues`, `SparseNumericDocValues`).

**Measurement:** `doc_values/varying_bpv` in `benches/hot_paths.rs`, over the
real `fixtures/data/doc_values_varying_bpv` segment — a Lucene-written NUMERIC
field that trips `Lucene90DocValuesConsumer.writeValues`'s `doBlocks` split.
Rust-only: what matters is the difference between access patterns on this side.

### O19 — the varying-bits-per-value block header was re-read for every value

`numeric_value` re-reads the per-field jump table and the block header (one
byte, one `i64`, one `i32`) on every call. Lucene's
`VaryingBPVReader.getLongValue` opens with `if (this.block != block)` and keeps
the decoded block, paying that once per 16,384-value block.

The first measurement was two numbers that were the same, and that was the
finding:

| access pattern | free function |
|---|---|
| `stride1` (16,384 consecutive calls per block) | 13.08 ns |
| `stride16k` (a block crossing on every call) | 13.09 ns |

An implementation that caches the block is much faster on the first and no
faster on the second. Identical timings mean nothing is cached.

Fixed by adding `NumericReader`, a cursor holding what Lucene's per-field
readers hold: the decoded block, and (for a sparse field) the `IndexedDISI`
doc-id list decoded once rather than per call. Same "give it an owner" fix as
`FieldNorms` above.

| access pattern | free function | `NumericReader` | |
|---|---|---|---|
| `stride1` | 13.08 ns | **8.16 ns** | 1.60x |
| `stride16k` | 13.09 ns | 16.82 ns | 0.78x |

**The second row is not hidden and is worth reading carefully.** When every
single call crosses a block, the cursor pays a cache-miss check and an
indirection for nothing, and is 29% slower. Lucene's reader has exactly the same
property — the same branch, the same trade — and accepts it because Lucene's
`NumericDocValues` is a `DocIdSetIterator`: forward-only by contract, so a
consumer *cannot* express the pattern `stride16k` measures. `numeric_value`
stays for single-lookup callers, and nothing is forced onto the cursor.

`decode_value_varying_bpv` and `NumericReader` now share one block-header parser
(`read_varying_block`), so there is one implementation of that wire format
rather than two to keep in step. The fixture test asserts the two paths agree
document-for-document against real Lucene-written data in three access orders —
forward, backward, and strided so that every call is a cache miss — plus
out-of-range behaviour, because a cache exercised only forwards can be wrong in
the other two directions.

### Not changed

`sorted_ord`, `sorted_numeric_values` and `binary_value` have the same
free-function shape and the same sparse-DISI cost. They are left alone: no
benchmark here exercises them, and the fix is the same `IndexedDISI` cursor
already filed above, which would remove the need to give each of them an owner
individually.

---

# The headline finding: we decode blocks Lucene never touches

Everything above is component work. This is the structural divergence the sweep
was for, and it reframes the rest.

**Method.** Timing could not settle this — see the noise-floor note below. Both
engines were instrumented to *count* instead: documents the scorer produced
(`collect()` per leaf), and, on this side, full block-body decodes
(`ForUtil`/`PForUtil` unpack of 256 documents). Counts are immune to
measurement noise and separate two very different defects: "we are slower" and
"we do more work".

## Documents scored, top-50 over 5M documents

| query | this port | Lucene | ratio | time |
|---|---|---|---|---|
| `q12` `or t0 t1 t2 t3` | 4,121,444 | 1,625 | **2536x** | 0.23x |
| `q09` `and t0 t1 t2` | 1,179,565 | 1,451 | **813x** | 0.23x |
| `q11` `or tz t2s` | 1,334,994 | 11,505 | **116x** | 0.15x |
| `q06` `and t0 t1` | 155,249 | 1,730 | **90x** | 0.31x |
| `q01` `term body:t0` | 82,564 | 1,425 | **58x** | 0.89x |
| `q20` `and title t0 t1` | 770,544 | 34,169 | 23x | 0.18x |

Hit sets match on all 20 queries, so this is the same query answered two ways.

**This reconciles the entire milestone.** Every per-document cost measured in
this sweep is now *better* than Lucene's — `ForUtil.decode` 2.17x, posting-list
`nextDoc()` 1.65x, `DirectReader.get` 1.81x. We are nonetheless 2-7x slower end
to end, because we do 58x-2500x more of it. The port is not slow. It is doing
work Lucene skips.

One instrumentation trap, recorded because it inverted the result: the first
run used `TopScoreDocCollectorManager(TOP_N, Integer.MAX_VALUE)`. That asks for
exact hit counts, which switches Lucene's block-max pruning **off**, and Lucene's
count came back as `maxDoc` for every term query -- making this port look 60x
*better*. `IndexSearcher.search(query, n)` uses a threshold of 1000; the counted
run has to use the same or it measures a different query than the timed one.

## Where the wasted work is: no `advanceShallow`

| query | blocks decoded | documents unpacked | documents actually scored | utilisation |
|---|---|---|---|---|
| `q07` `and t0 tz` | 23,702 | 6,067,712 | 80,226 | **1.3%** |
| `q10` `or t0 t1` | 38,728 | 9,914,368 | 138,650 | **1.4%** |
| `q06` `and t0 t1` | 38,728 | 9,914,368 | 155,249 | **1.6%** |
| `q01` `term body:t0` | 5,561 | 1,423,616 | 82,564 | 5.8% |
| `q09` `and t0 t1 t2` | 57,166 | 14,634,496 | 1,179,565 | 8.1% |

**We bit-unpack up to 99% of documents for nothing.**

Lucene separates the two operations explicitly. `Lucene104PostingsReader`:

```java
public void advanceShallow(int target) throws IOException {
  doAdvanceShallow(target);
  needsRefilling = true;          // impacts moved; block body NOT decoded
}
...
if (needsRefilling) { refillDocs(); needsRefilling = false; }   // decode, later
```

`ImpactsDISI.advanceTarget` loops on `advanceShallow` + `getMaxScoreForLevelZero`,
walking block after block on their *impacts alone*, and calls the underlying
`advance` -- which is what triggers `refillDocs` -- only once a competitive block
is found.

`LazyDocsCursor::advance` has no such split. Reaching a block decodes it, so the
scoring loop's "is this block competitive?" test is answered *after* paying the
`ForUtil` unpack it was supposed to avoid. The block-max pruning this port added
in M1.5 does skip blocks -- it just pays for them first.

That also explains the M1-e2e profile that started this whole milestone.
`decode_full_block_body` 9.5% + `decode_impacts` 9.2% + `ForUtil::decode` 6% was
read as "decode is a quarter of the query, spread thin". It was really "a quarter
of the query is decoding blocks we throw away".

## The fix

Give `LazyDocsCursor` the same split: an `advance_shallow(target)` that moves the
block position and decodes impacts only, a `needs_refilling` flag, and a
`refill()` at the point `next_doc`/`freq` first needs documents. Then the three
scoring loops consult the bound before the unpack instead of after.

Not attempted here. It restructures the cursor's state machine and every caller
of `level0_impacts`/`freq`, and it deserves its own measured milestone rather
than being bolted onto the end of this one. It is now the highest-value item
open, ahead of the block-tree work: the counters say the prize is up to 70x less
decode work on the conjunction and disjunction queries.

The instrumentation stays: `postings::test_only_block_decode_counter`,
`lucene_search::test_only_scored_docs_counter`, and the Java runner's counting
`LeafCollector`. Both runners now emit `scored` and `blocks` columns.

## Caveat on this milestone's smaller numbers

Investigating whether SIMD was worth adding turned up a measurement problem
worth recording. Running the **same binary against itself** three times, 2s
warmup + 3s measure per case, `for_decode` varies by a median of **1.21x**,
worst case 1.64x.

So differences in the 1.1x-1.4x range are not resolvable by this harness, and
two things in this document need reading with that in mind:

- The large findings are safe: `next_doc` 7.8x, `DirectReader` 3.7x at 64 bits,
  phrase 9x, the `IndexedDISI` scaling result (orders of magnitude), and the
  cumulative `for_util` 0.75x -> 2.17x. Each is far above the floor and each was
  reproduced across separate runs.
- The **step-by-step attributions** are softer than they are written. "O2 moved
  the median 1.47x -> 1.62x" and "O3 1.62x -> 2.26x" are at the edge of what the
  harness resolves. The endpoints are real; the individual steps should not be
  quoted as precise.

`scripts/bench-micro.sh` should interleave both engines within a run, repeat,
and refuse to report a difference smaller than the measured noise floor. Not
done here.

---

## MAXSCORE clause partitioning: implemented, measured, reverted

The `advanceShallow` fix removed the wasted *decoding*. It did not change how
many documents get *scored*, and that remains the larger divergence:

| query | this port | Lucene |
|---|---|---|
| `or t0 t1 t2 t3` | 4,121,444 | 1,625 |
| `or tz t2s` | 1,334,994 | 11,505 |
| `and t0 t1 t2` | 1,179,565 | 1,451 |

The obvious candidate was the missing clause partition -- MAXSCORE (Turtle &
Flood), which `WANDScorer` implements and which `try_disjunction_lazy`'s own
comment named as the thing it did not do. Implemented in full: a static
per-clause `max_score` from the term's statistics, clauses ordered by it, the
longest prefix whose maxima sum below the threshold declared non-essential,
candidates generated from essential clauses only, and non-essential clauses
advanced *to* each candidate rather than iterated.

**It is a net regression and is not in the tree.**

| query | before | after | |
|---|---|---|---|
| `or tz t2s` | 73.3 qps | 111.8 qps | **1.53x** |
| `or t0 t1` | 178.4 qps | 154.4 qps | 0.87x |
| `or t0 t1 t2 t3` | 11.4 qps | 6.9 qps | **0.61x** |

It wins exactly where the non-essential clauses can be *jumped over* and loses
where they cannot. On `or t0 t1 t2 t3` every term is frequent, so candidates stay
dense, and each one now costs a random `advance` into a huge posting list where
it previously cost a sequential `next_doc`. Scoring 11% fewer documents did not
pay for that.

One sub-finding worth keeping, because it was measured separately. The first
version bounded non-essential clauses by their *static* maxima -- sound, since a
clause not positioned on the span cannot have its block impacts trusted. That is
far looser than a block bound, the span skip stopped firing, and
`or t0 t1 t2 t3` went to 6.2 qps while scoring *more* documents than before
(4.5M against 4.1M). Shallow-advancing every clause to the candidate first --
a header and a few vints, no body decode -- restores the tight bound and got it
back to 6.9. Still a regression, but it isolates the two effects.

### What this rules out, and what it points at

Lucene scoring 1,625 documents where we score 4,121,444 is *not* explained by
the clause partition alone. `WANDScorer` partitions on **per-span** maxima
refreshed through `advanceShallow`, not on static ones, which makes the
non-essential set much larger inside any given span and -- the part that
matters -- makes the candidate stream sparse enough that advancing the
non-essential clauses is cheap. A static partition gets the bookkeeping without
the sparsity, which is the half that pays.

So the remaining gap is a per-span partition, not a partition. That is a larger
change than this attempt and it should not be started until there is a
measurement that can tell a 1.2x regression from noise -- see the harness caveat
above. Filed, with this attempt's numbers as the baseline to beat.

This is the fourth measured revert in this project's performance work
(header-only block skipping, the WAND attempt in M1.5, the `ImpactsDISI`-shaped
conjunction, and this). All four looked obviously right beforehand.
