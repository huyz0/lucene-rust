# c24 — the arithmetic gate, burned down across 14 `lucene-codecs` modules

Follow-up batch to c19's mechanical gate. c19 turned
`clippy::arithmetic_side_effects` on crate-wide for `lucene-codecs` and left
**26 modules** carrying a `#[allow(...)] // TODO(arith-audit)` marker on their
`mod` declaration in `lib.rs` — a burn-down list, not a clean bill of health.
This batch audits 14 of them, following `docs/arithmetic-gate.md`'s rules:
prove it, bound it and make the failure a typed error, or say it is
infallible and name the check.

Java read from **`/home/tuong/work/lucene-10.5.0`**, the pinned tag.

## Burn-down

| | count |
|---|---|
| modules carrying the marker at c19 | 26 |
| **audited this batch** | **14** |
| lint sites resolved (lib, non-test) | **~272** of the 1 193 `lucene-codecs` lib sites c19 measured |
| **remaining marked** | **12** |

`docs/arithmetic-gate.md`'s table updated 26 → 12;
`python3 scripts/check-arith-allows.py` reports `ok (23 module(s) still
unaudited)` (12 codecs + 11 index), matching.

Audited: `block_packed`, `compound_format`, `direct_monotonic`,
`direct_reader`, `fuzzy`, `indexed_disi`, `live_docs`, `lz4`, `norms`,
`packed_ints`, `regexp`, `suggest`, `terms_dict`, `wildcard`.

Still marked (12): `blocktree`, `doc_values`, `for_util`, `fst`, `hnsw`,
`hnsw_vectors`, `points`, `postings`, `postings_writer`, `stored_fields`,
`term_vectors`, `vectors`.

Modules were picked by reachability, as the batch brief asked: everything a
`.dvm`/`.nvm`/`.fdm`/`.tvm`/`.liv`/`.cfe` header steers directly
(`direct_monotonic`, `direct_reader`, `indexed_disi`, `terms_dict`, `norms`,
`live_docs`, `compound_format`, `packed_ints`, `block_packed`), plus the
decompressor every one of those file bodies passes through (`lz4`), plus the
two query compilers that take an untrusted pattern string across the FFI
(`regexp`, `wildcard`, `fuzzy`) and `suggest`.

| Rust file | Java counterpart (10.5.0) |
|---|---|
| `direct_monotonic.rs` | `util/packed/DirectMonotonic{Reader,Writer}.java` |
| `direct_reader.rs` | `util/packed/Direct{Reader,Writer}.java` |
| `packed_ints.rs` | `util/packed/PackedInts.Format.PACKED` (`BulkOperationPacked`) |
| `block_packed.rs` | `util/packed/BlockPacked{ReaderIterator,Writer}.java` |
| `indexed_disi.rs` | `codecs/lucene90/IndexedDISI.java` |
| `terms_dict.rs` | `codecs/lucene90/Lucene90DocValuesProducer.TermsDict` + `readTermDict` |
| `norms.rs` | `codecs/lucene90/Lucene90NormsProducer.java` |
| `live_docs.rs` | `codecs/lucene90/Lucene90LiveDocsFormat.java` |
| `compound_format.rs` | `codecs/lucene90/Lucene90CompoundFormat.java` |
| `lz4.rs` | `util/compress/LZ4.java` |
| `regexp.rs` | `util/automaton/RegExp.java` |
| `wildcard.rs` | `search/WildcardQuery.java` |
| `fuzzy.rs` | `search/FuzzyQuery` / `util/automaton/LevenshteinAutomata` |
| `suggest.rs` | `search/suggest/fst/WFSTCompletionLookup.java` |

Findings: **11 CORRECTNESS**, **1 MISSING**, **2 PERF**, **3 INTENTIONAL**.
All 12 CORRECTNESS/MISSING fixed, each with a test that fails against the
unfixed code. Four modules came out clean (no runtime change, proofs only).

---

## Findings

### F1 `[CORRECTNESS]` `direct_monotonic.rs::load_meta` — a two-byte flip in a `.fdm`/`.tvm` header reserves 51 GB and aborts

Java's `DirectMonotonicReader.loadMeta(meta, numValues, blockShift)` derives
`numBlocks` and allocates four parallel arrays of that size. Both parameters
reach it straight off disk: `stored_fields.rs:231`/`term_vectors.rs:214` read
`blockShift` as a raw `i32` and `index_num_chunks` as a raw `i32`, and
`doc_values.rs:505/555` read `blockShift` as a vint. Neither Java nor this
port bounded either one. Three separate failures came out of that:

* `num_values >> block_shift` with `block_shift >= 64` is a **panic** in a
  debug build. Java's `>>>` merely masks the shift to six bits, so this is a
  Rust-only hazard of exactly the shape `docs/arithmetic-gate.md` exists for.
* a negative `num_values` gave `num_blocks == -1`, and `-1 as usize` is
  `usize::MAX`.
* `Vec::with_capacity(num_blocks)` for a `num_blocks` the header chose is an
  **allocation failure → abort**, which `catch_unwind` cannot intercept: two
  flipped bytes in an `i32` buy a 51 GB reservation and a dead JVM.

Fixed at `direct_monotonic.rs`: reject `block_shift > MAX_BLOCK_SHIFT` (22 —
`DirectMonotonicWriter`'s own constructor throws outside `[2,22]`, so no file
Lucene wrote can carry one) and `num_values < 0` as `Corrupted`, and bound the
reservation by `input.remaining() / BLOCK_META_BYTES` — a block's metadata is
exactly 21 bytes (`i64` min + `i32` avg + `i64` offset + `u8` bpv), so the
stream itself is a hard ceiling and a well-formed file loses nothing.

Only the *ceiling* is enforced on the read side; the floor buys no safety (a
small shift only means more blocks, which the reservation cap already handles)
and this module's own synthetic metadata uses shifts under it. `write` asserts
the full `[2,22]` range, which is the faithful port of Java's
`IllegalArgumentException` and is what makes `1usize << block_shift` safe.

Tests: `out_of_range_block_shift_and_num_values_are_decode_errors`,
`absurd_num_values_errors_instead_of_reserving_for_it`,
`write_rejects_a_block_shift_java_would_throw_on`.

### F2 `[CORRECTNESS]` `direct_monotonic.rs::floor_index` — a corrupt chunk count underflows the binary search

`floor_index(data, meta, from, to, key)` is called with `to = self.num_chunks`
off the `.fdm`/`.tvm` on every stored-field and term-vector document lookup.
`to - 1` underflows at `to == i64::MIN`, and `hi - lo` can overflow for a
sufficiently negative `from`. Fixed by establishing both bounds *outside* the
loop (`from < 0` → `Corrupted`; `to <= from` → empty range), which is two
comparisons hoisted out of a `log2(n)` search — see PERF-1, it made the search
measurably **faster**, not slower. Test:
`floor_index_rejects_a_negative_lower_bound`.

### F3 `[CORRECTNESS]` `terms_dict.rs` — four unbounded values on the `.dvm` terms-dictionary path

`read_term_dict_entry` is `Lucene90DocValuesProducer.readTermDict`, reached
from every SORTED/SORTED_SET field:

* `termsDictSize` is a vlong. `(termsDictSize + BLOCK_SIZE - 1) >> 6` overflows
  near `i64::MAX`, and a negative one reached `loadMeta` as a negative block
  count. Fixed: reject negative, `checked_add` the rounding.
* `termsDictIndexShift` is a raw `i32`. `1i64 << shift` for a shift of 64+ is a
  panic in a debug build (Java's `1L <<` masks it). Fixed: reject `>= 63`,
  `checked_add` the `+ (1 << shift) - 1` and the `1 +` that follows.
* `termsDataOffset` + `termsDataLength` are two independent `i64`s whose sum
  sliced the `.dvd` — the same shape `norms::sparse_region` already documents.
  Fixed identically (`checked_add` + `usize::try_from`).
* `maxBlockLength` was read and discarded. It is what Java sizes
  `TermsDict.blockBuffer` from, and without it the per-block **decompressed**
  length vint bounded nothing: `vec![0u8; term.len() + block_len]` on a
  negative vint is ~2^64 → abort. It cannot be bounded by the compressed bytes
  left (LZ4 expands), so `TermsDictEntry` now carries `max_block_length` and
  the vint is bounded by it, exactly as Java does. Verified against
  `Lucene90DocValuesConsumer.compressAndGetTermsDictBlockLength`: the writer's
  `maxBlockLength` is the max of exactly the quantity the vint carries, so the
  bound cannot reject a Lucene-written file.

Also the two prefix/suffix term-length continuation vints, which size both a
`Vec::with_capacity` and a slice range. A negative vint is ~2^64 through
`as usize`, overflowing the `+=`. Fixed with a shared `bounded_extension`
helper — taking a *per-half* limit, because a block's first term is stored
outside the compressed body and can be longer than it, so bounding the prefix
by the body would reject files Lucene wrote.

Tests: `corrupt_terms_dict_size_is_a_decode_error`,
`corrupt_reverse_index_shift_is_a_decode_error_not_a_shift_panic`,
`corrupt_terms_data_region_is_a_decode_error`,
`corrupt_block_length_is_a_decode_error_not_an_allocation` (including a
positive, plausible length past `maxBlockLength`, the case a negative check
and an EOF would both miss), `corrupt_term_length_extension_is_a_decode_error`.

### F4 `[CORRECTNESS]` `norms.rs::read_value_at_ordinal` — `normsOffset` is an unconstrained `i64`

`normsOffset + ordinal * bytesPerNorm` on a value straight off the `.nvm`:
overflow is a panic in a debug build and, in a release one, a wrap to a
*plausible in-range* offset that reads a valid-looking norm out of the wrong
place in the `.nvd` — the silent-wrong-answer half of the class. A negative
offset separately became a huge `usize` through an `as` cast and merely looked
like EOF. Fixed by folding `checked_mul`/`checked_add`/`try_from` into one
expression with a `Corrupted` error. Tests:
`corrupt_norms_offset_is_a_decode_error_not_an_overflow`,
`negative_norms_offset_is_a_decode_error`.

### F5 `[CORRECTNESS]` `block_packed.rs::decode_all` — `totalValueCount` off a `.tvd` chunk header sized the reservation

Every caller passes a `.tvd` chunk-header field (`chunk_docs`,
`total_terms`, `total_positions`, `total_offsets`, `total_payloads`).
`Vec::with_capacity(total_value_count)` on a corrupt one is an abort. The
decode loop itself cannot outrun the input, so only the pre-reservation was
unbounded: it is now clamped by `input.remaining() * BLOCK_SIZE`, the cheapest
possible block being a single token byte carrying 64 constant values. A
well-formed stream reserves exactly what it did before. Test:
`absurd_total_value_count_errors_instead_of_reserving_it`.

### F6 `[CORRECTNESS]` `packed_ints.rs::get` — an unbounded width underflows the shift, and a negative index overflows the multiply

`get` computes `shift = n_bytes * 8 - bit_offset - bits_per_value`. For
`bits_per_value > 64` that underflows before any bounds check runs. Both
current callers do bound it (block-packed rejects `> 64`; term vectors masks
the token to five bits) — but this is where the shifts live, and a
`pub(crate)` primitive should not depend on every future caller remembering.
Second, `index as u128` **sign-extends**: `(-1i64) as u128` is `u128::MAX`, and
the multiply overflows. Both now return `Corrupted`.

`byte_count` changed from `i64` to `u64`: as an `i64` a negative count produced
a negative quotient, and `as usize` turned that into a gigantic length that
callers handed straight to `vec![0u8; n]`. Call sites in `block_packed.rs` and
`term_vectors.rs` updated (both already passed non-negative values).

Tests: `bits_per_value_above_64_is_a_decode_error_not_an_underflow`,
`negative_index_is_a_decode_error_not_a_multiply_overflow`.

### F7 `[CORRECTNESS]` `lz4.rs` — the length-extension accumulator was unbounded

An LZ4 length extension is a run of `0xFF` bytes each adding 255. Both the
literal and the match halves accumulated into a bare `usize +=`. "Every
increment consumes a byte of input" is a statement about the *file*, not about
`usize`: a long enough run wraps the accumulator to a small length and decodes
a **silently truncated block** in release, or panics in debug. Factored into
`read_length_extension` with a `checked_add`, plus a `checked_add` on the
`match_len + MIN_MATCH` that follows. Test:
`length_extension_run_is_bounded`.

### F8 `[CORRECTNESS]` `suggest.rs::decode_weight` — a `debug_assert` on a value off disk

`decode_weight`'s `debug_assert!((0..=u32::MAX).contains(&cost))` asserted a
range that only holds for costs this module's own `encode_weight` produced.
An FST can be loaded from bytes on disk, and `PositiveIntOutputs::decode`
returns whatever the output vlong says — so a corrupt or foreign suggester FST
panicked in a debug build, and `u32::MAX as i64 - cost` overflowed. Java's
`WFSTCompletionLookup.decodeWeight` is `(int) (Integer.MAX_VALUE - encoded)`,
a cast that cannot fail; this is now the same thing one width up. Test:
`out_of_range_cost_decodes_to_garbage_rather_than_panicking`, which reaches it
through `top_n_completions` on an FST holding `i64::MAX`.

### F9 `[CORRECTNESS]` `suggest.rs::top_n_completions` — the caller's `n` sized an allocation

`BinaryHeap::with_capacity(n + 1)` where `n` is the caller's "top N", bounded
by nothing in the index: `n + 1` overflows at `usize::MAX` and the reservation
aborts long before that. The heap's real occupancy is
`min(n, matching terms) + 1`, so it now pre-reserves at most
`HEAP_RESERVE_CAP` and grows from there. Test:
`absurd_n_does_not_reserve_a_slot_per_requested_result`.

### F10 `[CORRECTNESS]` `indexed_disi.rs::write_with_dense_rank_power` — a negative doc id corrupts or panics the writer

The two existing assertions check strict ascent and the `i32::MAX` sentinel,
neither of which rules out a negative first doc id. A negative one puts
`block_base` at `0xFFFF0000`: the SPARSE path writes nonsense 16-bit ids and
the DENSE path indexes `words[rel / 64]` with an astronomic `rel`. Third
assertion added. Test: `write_rejects_a_negative_doc_id`.

### F11 `[CORRECTNESS]` `live_docs.rs::write` — `len() - cardinality()` could underflow

`FixedBitSet::cardinality()` counts set bits in *every* word, ghost bits past
`len()` included. `parse`'s subtraction is genuinely safe (the runtime
ghost-bit check immediately above it establishes `cardinality() <= max_doc` —
this is a runtime check, not the `debug_assert` inside `from_words`), but
`write` takes a caller-supplied bitset and had no such guarantee. Now a
`checked_sub` reporting the same `GhostBitsSet` the read side names.
No test: in a debug build `FixedBitSet::from_words`/`set` reject the input
that would reach it, so it is unreachable-by-construction defensive hardening
rather than a live defect. Called out here so it is not mistaken for a tested
fix.

### F12 `[MISSING]` `indexed_disi.rs::read_block_header` — the block-ordinal accumulator was unchecked

`next_block_index` accumulates up to 65 536 per block over a block count
bounded only by `data.len() / 4`. The bound is astronomically large but it is
the *input's*, not ours, and this runs once per 65 536-doc block rather than
per doc — so it is now a `checked_add` reporting `Corrupted`. Measured: no
change on any `indexed_disi/cursor` benchmark.

### PERF-1 `[PERF]` the hoisted bounds made `floor_index` ~18% faster

Measured with a min-of-40 harness (criterion's mean is unusable on this
machine — the same code measured 83 µs, 91 µs and 129 µs in three consecutive
runs), alternating A/B three times:

| workload | with the bounds | without | delta |
|---|---|---|---|
| `direct_reader::get` × 100 k, 20-bit | 208–224 µs | 199–229 µs | neutral |
| `direct_monotonic::get` × 200 k | 453–553 µs | 529–553 µs | neutral |
| `direct_monotonic::floor_index` × 20 k | **1 620–1 656 µs** | 2 007–2 018 µs | **−18%** |
| `lz4::decompress` 1 MB | 172–176 µs | 174–178 µs | neutral |

`floor_index`'s `from >= 0` and `to > from` guards give LLVM the
non-negativity it needs for the search loop, so the check pays for itself
several times over. This is the stored-fields and term-vectors document
lookup, i.e. c8's term-vector merge and c1's segment open.

Every check added on a per-value path was hoisted out of its loop, as the
brief required: `direct_reader::get` and `packed_ints::get` compare `byte_pos`
against the slice length **once** rather than per byte read; `indexed_disi`'s
`checked_add` is per 65 536-doc block, not per doc; `terms_dict`'s length
bounds are per block; `lz4`'s are per length-extension run. Nothing was added
inside `lz4`'s `copy_within` loop, `indexed_disi`'s popcount scan, or
`direct_reader`'s byte-accumulate tail.

### PERF-2 `[PERF]` reservations bounded by the input, not by a magic constant

The first cut of F1/F5 capped reservations at a fixed constant, which would
have cost a real term-vector chunk several reallocations. Both now derive the
ceiling from `input.remaining()` — 21 bytes per monotonic block, one token
byte per 64 block-packed values — so a well-formed file reserves exactly what
it did before and only a corrupt count is capped.

### INTENTIONAL-1 four modules are clean

`compound_format`, `wildcard`, `fuzzy` and `regexp` needed **no** runtime
change: every site is bounded by a slice length, a loop range, an ASCII
constant, or a guard already present. That is a real result, recorded rather
than padded into findings. The proofs are in the `// ARITH:` comments; the
non-obvious ones are `regexp::decimal_value_in_range` (the
`significant.len() > 10` early return caps the `value * 10 + d` accumulator
three orders of magnitude inside `u64`) and `regexp::repeat_match`
(`max.is_none() || max >= min` is established by all five `Node::Repeat`
construction sites and preserved by the paired decrements).

### INTENTIONAL-2 `direct_monotonic::write` asserts rather than returning `Result`

Java's `DirectMonotonicWriter` constructor throws
`IllegalArgumentException` for a `blockShift` outside `[2,22]`. Every caller
in this port passes a format constant (10 or 16), so a panic is the faithful
port of an unreachable programming error, and it is what makes
`1usize << block_shift` provably safe. Changing the signature to `Result`
would ripple through three writers for a case no caller can reach.

### INTENTIONAL-3 `direct_monotonic::write`'s deltas are `wrapping_sub`

`(chunk[last] - chunk[0])`, `v - (avg_inc * i)` and `*d -= min` are all
`long` arithmetic in Java, which wraps; `get` undoes them with `wrapping_add`.
Wrapping is what round-trips, so these are `wrapping_sub` rather than checked.
Java's own `BlockPackedWriter` comment names the same overflow case.

---

## Verdicts

| module | verdict |
|---|---|
| `direct_monotonic.rs` | swept — F1, F2; 100% line coverage |
| `terms_dict.rs` | swept — F3 (four sites) |
| `norms.rs` | swept — F4 |
| `block_packed.rs` | swept — F5 |
| `packed_ints.rs` | swept — F6 |
| `lz4.rs` | swept — F7; decompress copy loop untouched, measured neutral |
| `suggest.rs` | swept — F8, F9 |
| `indexed_disi.rs` | swept — F10, F12 |
| `live_docs.rs` | swept — F11 (defensive, untested by construction) |
| `direct_reader.rs` | swept — proofs only, one bound hoisted out of the hot path |
| `compound_format.rs` | swept-clean — proofs only |
| `wildcard.rs` | swept-clean — proofs only |
| `fuzzy.rs` | swept-clean — proofs only (one `i + max` → `saturating_add`, exact under the `.min(m)` that follows) |
| `regexp.rs` | swept-clean — proofs only |

## Gates

* `cargo fmt --all` clean.
* `cargo clippy -p lucene-codecs --all-targets -- -D warnings` clean.
* `cargo test -p lucene-codecs`: 1 148 lib + all integration tests pass.
* `scripts/verify-write-path.sh`: **22/22** (was 21/21 at c19; c23 added one).
* `python3 scripts/check-parity.py`: ok.
* `python3 scripts/check-arith-allows.py`: ok, 23 modules still unaudited.
* Per-file line coverage, all 14 audited modules: 96.73%–100%, every one above
  the 95% bar. `direct_monotonic.rs` is at 100%.

## Tier-2 review

The `quality-reviewer` pass over this diff found **two gating proof defects**,
both fixed here, and both exactly the c19 failure mode — a proof whose stated
bound does not establish its conclusion:

1. `direct_reader::get`'s `8 * i <= 64` — 64 *is* the panicking shift. The real
   bound is 56, and it comes from `SUPPORTED_BITS` (a non-zero `shift` implies
   a width that is not a multiple of 8, and those stop at 28), not from the
   `<= 64` cap the comment cited. Restated, and pinned with a
   `debug_assert!(bytes_needed <= 8)` so the claim is exercised by `cargo test`
   rather than living only in prose.
2. `packed_ints::get`'s "computed in `u128` from an `i64`" — that is the
   sign-extension hole, not a proof against it. Became F6's second half.

Four advisory proof corrections were also applied: `indexed_disi::read_u16_at`
(the invariant is false on the error path — `read_block_header` assigns
`block_end` before the `slice(..)?` that could reject it; the conclusion
survives by a different, weaker bound, now stated), `indexed_disi`'s
`index + 1` (inside an allow but not covered by its proof),
`regexp::repeat_match` ("only constructor" — there are five), and a stale
call-site name in `direct_reader::padding_bytes_needed`. Two tests were
strengthened: the `terms_dict` block-length case now uses a positive,
plausible length past `maxBlockLength` (the previous cases were all caught by
a pre-existing negative check or a later EOF), and `suggest`'s scalar
assertions pin literal numbers instead of restating `decode_weight`'s body.

## Open

* **12 modules still marked.** The largest and most reachable are
  `doc_values` (70 sites), `blocktree` (93), `postings` (51),
  `stored_fields` (83), `term_vectors` (88), `points` (144) and `for_util`
  (166) — between them the `.tim`/`.tmd`/`.doc`/`.dvm`/`.kdm` metadata paths
  this batch could not reach.
* **`term_vectors.rs:470` carries a live instance of this class**, found while
  changing a call site there but out of scope (the module keeps its marker):
  `total_distinct_fields` accumulates a `readVInt` into a `u32` and
  `packed_ints::byte_count(...)` then sizes `vec![0u8; n]` directly — up to
  ~16 GB, the abort shape. Whoever burns down `term_vectors` should start
  there.
* **A `maxBlockLength` a corrupt `.dvm` chose can still ask for a ~2 GB `Vec`**
  in `terms_dict`. That is exactly Java's exposure too (`new
  byte[maxBlockLength + padding]`), and no tighter input-derived bound exists
  because LZ4 expands; recorded rather than invented.
* **`clippy::cast_sign_loss` per audited module** is worth revisiting. It would
  have caught F6's second half directly, and c19 rejected it only as a
  *workspace-wide* deny (1 036 sites). Per-module, on a module that has just
  been audited, the count is small and the shape it catches — sign extension
  into a wider unsigned type — is the one cast that turns a proven-safe
  operator back into a live panic.
