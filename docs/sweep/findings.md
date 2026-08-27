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
