# m2 sweep — `c2-sparse-lookup`

Follow-up batch opened from `b6-docvalues` #9 and `b2-packed` #16/#17/open-item-3.
Java source of truth: `/home/tuong/work/lucene` (Lucene 10.5.0).

Files swept:

- `crates/lucene-codecs/src/indexed_disi.rs`
- `crates/lucene-codecs/src/doc_values.rs` (the reader/writer surfaces that
  compose an `IndexedDISI` — `NumericReader`, `numeric_value`, `binary_value`,
  `sorted_ord`, `sorted_numeric_values`, `write_sparse_numeric_entry_body`, the
  sparse BINARY writer)
- `crates/lucene-codecs/src/norms.rs` (`norm_value`, the sparse norms writer)
- `fixtures/src/VerifySparseNumericDocValues.java`
- `crates/lucene-codecs/benches/hot_paths.rs`

---

## `crates/lucene-codecs/src/indexed_disi.rs`

Java counterpart: `codecs/lucene90/IndexedDISI.java` (whole class), plus
`codecs/lucene90/Lucene90DocValuesProducer.getNumeric`/`SparseNumericDocValues`/
`DenseNumericDocValues` for how the cursor is composed with a `LongValues`.

### Method correspondence

| Rust | Java | Verdict |
|---|---|---|
| `dense_rank_bytes` | ctor + `writeBitSet` `denseRankPower` validation, `denseRankTable` length | identical (ported by b2) |
| `decode_doc_ids` | `readBlockHeader` + the three `Method` bodies, driven to exhaustion | not-in-Java by design; kept for the callers that want an owned doc-id list (see #8) |
| `DisiCursor::new` | ctor | identical, minus `prefetch` (no `IndexInput` here — the region is already a `&[u8]` over mmap) and minus the jump table (#7) |
| `DisiCursor::advance_exact` | `advanceExact` | identical structure (`block < targetBlock` → `advanceBlock`, then `block == targetBlock && advanceExactWithinBlock`, then `doc = target`), plus the sentinel guard (#6) and the forward-only assert (#2) |
| `DisiCursor::advance_block` | `advanceBlock`'s iteration fallback (`do { seek(blockEnd); readBlockHeader(); } while (block < targetBlock)`) | identical; the jump-table shortcut is still absent (#7) |
| `DisiCursor::read_block_header` | `readBlockHeader` | identical field for field (`block`, `index = nextBlockIndex`, `nextBlockIndex = index + numValues`, per-method `blockEnd`/`gap`/`denseBitmapOffset`/`wordIndex = -1`/`numberOfOnes = index + 1`/`denseOrigoIndex`), plus a range check per block payload |
| `DisiCursor::slice` | *(none — Java's `IndexInput` throws on read)* | not-in-Java; one bounds check per block instead of one per read |
| `DisiCursor::advance_exact_within_block` | `Method.advanceExactWithinBlock` dispatch | identical |
| `DisiCursor::sparse_advance_exact` | `Method.SPARSE.advanceExactWithinBlock` | identical, including the `nextExistDocInBlock > targetInBlock` early-out, the `target == doc` answer cache, and the `index--` / `seek(fp - 2)` push-back on an overshoot |
| `DisiCursor::dense_advance_exact` | `Method.DENSE.advanceExactWithinBlock` | identical, including the `targetWordIndex - wordIndex >= 1 << (power - 6)` rank-skip test and `index = numberOfOnes - bitCount(word >>> target)` |
| `DisiCursor::rank_skip` | `rankSkip` | identical (`rankIndex = targetInBlock >> power`, big-endian 2-byte entry, `rankAlignedWordIndex = (rankIndex << power) >> 6`, `numberOfOnes = denseOrigoIndex + rank + bitCount(rankWord)`) |
| `DisiCursor::reset` | *(none — Java constructs a new `IndexedDISI`)* | added (#2) |
| `DisiCursor::doc_id` | `docID()` | added |
| `write` / `write_with_dense_rank_power` | `writeBitSet(it, out)` / `writeBitSet(it, out, denseRankPower)` | rank table now written (#4); jump table still not (#7) |
| `create_rank` | `createRank` | added (#4) |
| `rank_of` | — | not-in-Java helper over `decode_doc_ids`' output |
| — | `advance`, `nextDoc`, `intoBitSet`/`intoBitSetWithinBlock`, `docIDRunEnd`, `index()`, `cost()`, `asDocIndexIterator`, `createBlockSlice`/`createJumpTable`, `addJumps`/`flushBlockJumps` | **no Rust counterpart.** `advance`/`nextDoc` are the "seek to the next present doc" half of `DocIdSetIterator`; this port's doc-values API is random-access-by-doc-id, so only `advanceExact` has callers. `intoBitSet`/`docIDRunEnd` are bulk/SIMD APIs over an iterator this port does not expose. The jump-table trio is #7. |

### Findings

1. **[PERF] `ordinal_within_block` recomputed the rank from the block start on
   every call.** Java's `IndexedDISI` carries `word`, `wordIndex`,
   `numberOfOnes` and `denseOrigoIndex` across `advanceExact` calls, so a
   forward walk reads each of a DENSE block's 1024 words exactly **once for the
   whole block**. Us: every lookup seeked to the target word and then re-read
   and popcounted every word before it — up to 1024 `i64` reads *per lookup*,
   which makes a forward scan of a DENSE block quadratic in its cardinality.
   The SPARSE branch was worse in the same way: it rescanned the block's 16-bit
   doc ids from `payload_start` on every call, where Java keeps the slice
   pointer, `index`, `exists` and `nextExistDocInBlock`.
   **Fixed** — `DisiCursor` is now a field-for-field port of Java's state (see
   the table above). Measured on `indexed_disi/cursor` in `benches/hot_paths.rs`
   (10,000 in-order lookups over a DENSE block, 4,000 over a SPARSE one):

   | arm | before | after | |
   |---|---|---|---|
   | `dense_forward/n10000` | 2.2151 ms | 37–40 µs | **~56x** |
   | `sparse_forward/n4000` | 6.4182 ms | 12–15 µs | **~450x** |

   Per lookup that is 221 ns → ~4 ns (DENSE) and 1604 ns → ~3.5 ns (SPARSE).
   (Ranges, not point estimates — see the benchmark caveat at the end.)
   Tests: `cursor_matches_decode_for_a_{sparse,dense,all}_block`,
   `cursor_matches_decode_across_block_boundaries`,
   `cursor_matches_decode_for_an_all_block_followed_by_a_dense_one` (an ALL
   block contributes 65,536 to the next block's ordinal base while writing zero
   payload bytes — the combination that breaks a cursor deriving its base from
   bytes consumed), each run at `NO_RANK` and powers 7/9/12/15 and each in two
   passes (every doc id, then only the present ones, which is the stride that
   reaches `rank_skip`).

2. **[CORRECTNESS] `advance_exact` answered `None` for a backward doc id.**
   Java's `advanceExact` requires a non-decreasing target and enforces it with a
   bare `assert` — off in production, undefined behaviour when violated. Us: a
   silent `Ok(None)`, which is *the worst of the three options*, because "this
   document has no value" is a legitimate answer: a caller that violated the
   contract got a plausible wrong number instead of a diagnosis, and the old
   comment even advertised it ("says so by returning `None`").
   **Fixed** — a backward or negative doc id now **panics**, with the message
   naming both doc ids and pointing at `reset()`. Decision and rationale, since
   the task left it open: a panic, not an `Err`. This file already panics on a
   violated *writer* contract (`write`'s strictly-ascending assert), a violated
   reader precondition is the same class of caller bug, and `lucene_store::Error`
   has no variant that means "you called this wrong" — `Corrupted` would blame
   the data. Corrupt bytes stay an `Err`, as everywhere else in this port.
   To keep random access possible, `DisiCursor::reset()` rewinds to the start of
   the region (Java's equivalent is constructing a new `IndexedDISI`); it
   allocates nothing and keeps the same borrow. `doc_id()` (Java's `docID()`)
   lets a caller decide when to call it — which is exactly what `NumericReader`
   now does. Tests:
   `cursor_panics_on_a_backward_doc_rather_than_answering`,
   `cursor_panics_on_a_negative_doc`,
   `reset_restores_random_access_and_agrees_with_decode` (5,000 pseudo-random
   probes, present and absent, each checked against `decode_doc_ids` +
   `rank_of`), `repeating_the_same_doc_returns_the_same_answer` (Java's
   `target == doc` cache, for both a present and an absent doc).

3. **[MISSING] The DENSE rank table was parsed past but never used.** Java's
   `rankSkip` turns a cold lookup deep inside a DENSE block from "popcount every
   word before the target" into "one 2-byte rank read plus at most
   `2^(power - 6)` words". Real Lucene writes a rank table on **every** DENSE
   block (`writeBitSet`'s two-argument form defaults to
   `DEFAULT_DENSE_RANK_POWER = 9`), so this was dead weight in every real
   segment this port reads.
   **Fixed** — `DisiCursor::rank_skip`, a line-for-line port including the
   `(rankIndex << power) >> 6` alignment and the absolute-not-relative
   `denseOrigoIndex` base. Measured (`indexed_disi/cursor/dense_random*`, a
   fresh cursor per lookup at doc 54,000 of a 10,000-present DENSE block):
   **~420 ns without a rank table → ~16 ns with one, ~26x.**
   Tests: `rank_skip_produces_the_same_ordinals_as_a_full_walk` (strides large
   enough that *every* answer goes through `rank_skip`, at all four legal
   powers), plus the rank-power sweep inside `assert_cursor_matches` (#1).

4. **[MISSING] `createRank` was not ported, so this port's own sparse fields
   could never use #3.** b2 recorded this as not-a-compatibility-defect
   (`denseRankPower = 0xFF` in our metadata is Java's own "no table" encoding,
   so a real Lucene reader never looks for one) and left it. That reasoning is
   still right about *correctness* and wrong about *cost*: it left every sparse
   doc-values and norms field this port writes ~26x slower to random-access than
   the same field written by Lucene, for 256 bytes per DENSE block.
   **Fixed** — `create_rank` ports `createRank`, `write_with_dense_rank_power`
   ports the three-argument `writeBitSet`, and the three sparse write sites
   (`doc_values::write_sparse_numeric_entry_body`, the sparse BINARY writer,
   `norms::write_fields`' sparse branch) now emit the table at
   `DEFAULT_DENSE_RANK_POWER` and record `9` in the metadata byte, as
   `Lucene90DocValuesConsumer`/`Lucene90NormsConsumer` do.
   Tests: `create_rank_matches_javas_byte_layout` (the exact big-endian
   before-this-sub-block counts pinned by hand, not round-tripped — a
   self-consistent-but-not-Java table would round-trip fine and be silently
   wrong for a real Lucene reader), `write_with_dense_rank_power_round_trips_
   through_decode`, `write_rejects_an_out_of_range_dense_rank_power`.
   **Differential**: `fixtures/src/VerifySparseNumericDocValues.java` gained
   three strided passes (strides 701 / 4096 / 20011). This matters: Java only
   consults the rank table when the target is at least `2^(power-6)` words
   ahead, so the pre-existing doc-by-doc pass never read a single rank byte.
   Negative control run and confirmed — perturbing one `bit_count` inside
   `create_rank` leaves the doc-by-doc pass green and fails the strided one
   (`MISMATCH (stride 701) _1 doc 189270: expected=1324887 got=1324908`).
   `scripts/verify-write-path.sh`: 14/14 against real Lucene 10.5.0.

5. **[CORRECTNESS] A truncated region read as "no value" instead of an error.**
   The old cursor set an `exhausted` flag when it ran out of bytes and answered
   `Ok(None)` from then on. But a well-formed region *always* ends with the
   `NO_MORE_DOCS` sentinel block, whose block index (`0x7FFF`) outranks every
   legal target — so the walk can only run off the end if the region is
   truncated, and that is corruption, not absence.
   **Fixed** — `read_block_header` returns `Err(Eof)`; `exhausted` is gone.
   Test: `a_truncated_region_is_an_error_not_a_missing_value` (the surviving
   first block still answers, the missing sentinel errors).

6. **[INTENTIONAL] `advance_exact(NO_MORE_DOCS)` returns `None` where Java
   returns `true`.** Java's sentinel block is a real SPARSE block holding
   `0xFFFF`, so `advanceExact(Integer.MAX_VALUE)` would report it present at the
   field's cardinality. No Java caller does this (they stop at `maxDoc`) and no
   caller here can either, but a decoded sentinel is a wrong answer that costs
   one comparison to rule out. Kept, documented, and tested
   (`the_no_more_docs_sentinel_is_never_reported_as_present`).

7. **[MISSING, not fixed] The block jump table.** Java appends an
   `(index, offset)` `int` pair per block and `advanceBlock` uses it when the
   destination is two or more blocks ahead. Us: block headers are walked.
   **Recorded** — the cost is `O(maxDoc / 65536)` four-byte reads, 16 of them
   for a million documents, against a jump table's one random read; and this
   port's writers record `jumpTableEntryCount = 0` (Java's own "no table"
   encoding, which Java itself emits whenever there is a single real block), so
   for our own files there is nothing to read. Worth revisiting only alongside a
   `nextDoc`-shaped iterator API, which this port does not have.

8. **[INTENTIONAL] `decode_doc_ids` stays.** It is no longer on any doc-values
   or norms lookup path, but two `lucene-search` callers genuinely want an owned
   doc-id list (`soft_deletes` builds a set from it; `field_norms` caches one —
   see #12), and it is the independent second implementation every cursor test
   asserts against. Keeping two decoders of one format is normally the shape
   that diverges silently; here the whole point is that they are checked against
   each other over every doc id in range, at every legal rank power.

9. **[PERF, recorded] A cold random DENSE lookup on a *rank-less* region got
   ~8% slower.** 392 ns → ~420 ns (`indexed_disi/cursor/dense_random`, a fresh
   cursor per lookup): the new code popcounts words `0..=target` where the old
   one seeked straight to the target word and counted the same words afterwards,
   and it now maintains four more state fields. Same asymptotics, slightly more
   bookkeeping. **Not fixed, and not worth fixing**: this arm only exists for
   regions written *without* a rank table, which after #4 means only files
   written by an older build of this port — every real Lucene file and every
   file this port now writes takes the ~16 ns rank path instead. Reported
   rather than buried because the batch's own rule was to show both numbers.

### Verdict

Swept; #1–#6 fixed with tests and (for #4) real-Lucene differential coverage
with a negative control. Open: #7 (jump table, recorded with its cost), #9
(recorded with numbers), #8 intentional.

---

## `crates/lucene-codecs/src/doc_values.rs`

Java counterparts: `codecs/lucene90/Lucene90DocValuesProducer` (`getNumeric`,
`SparseNumericDocValues`, `DenseNumericDocValues`, `getBinary`, `getSorted`,
`getSortedNumeric`) and `Lucene90DocValuesConsumer` (the sparse branches of
`addNumericField`/`addBinaryField`).

Only the `IndexedDISI`-composing surfaces were in scope this batch; b6 swept the
file as a whole.

### Method correspondence

| Rust | Java | Verdict |
|---|---|---|
| `NumericReader::new` / `value` | `getNumeric` composing an `IndexedDISI` with a `LongValues` (`SparseNumericDocValues.advanceExact` + `values.get(disi.index())`) | was divergent (O(cardinality) `Vec`) → **fixed** (#10); random access is a deliberate superset (#11) |
| `numeric_value` | `getNumeric` + `advanceExact`, one-shot | identical; fresh cursor per call by design (it is the single-lookup entry point) |
| `binary_value` | `getBinary`'s sparse branches | identical, cursor-backed already |
| `sorted_ord` | `getSorted().ordValue()` | identical (delegates to `numeric_value`) |
| `sorted_numeric_values` | `getSortedNumeric`'s two branches | identical, cursor-backed already |
| `write_sparse_numeric_entry_body`, the sparse BINARY writer | `addNumericField`/`addBinaryField`'s `numDocsWithValue != maxDoc` branch → `IndexedDISI.writeBitSet(it, data)` | rank table was missing → **fixed** (indexed_disi #4) |

### Findings

10. **[PERF] `NumericReader` held an O(cardinality) `Vec<i32>` of every present
    doc.** This is the carry-over b6 raised and could not fix. Java composes an
    `IndexedDISI` cursor with a `LongValues` and holds **no per-doc array at
    all**; we decoded the whole docs-with-field region at construction so we
    could binary-search it.
    **Fixed** — the reader holds a `DisiCursor`, which is 128 bytes inline
    (`NumericReader` is 200 bytes in total) and allocates nothing.

    Formula. Before: `4 bytes × Vec capacity`, and the capacity is the doubling
    sequence's next step at or above the cardinality — so up to **8 bytes per
    present doc** at steady state, and up to **12 bytes per present doc** at the
    instant of the last reallocation, when the old and new buffers are both
    live. After: **0 bytes of heap, independent of cardinality.**

    Measured with a counting global allocator, 1,000,000 documents:

    | density | present docs | before, steady | before, peak | before, total allocated | after |
    |---|---|---|---|---|---|
    | 1 % | 10,000 | 65,536 B | 98,304 B | 131,056 B in 13 allocations | **0 B, 0 allocations** |
    | 10 % | 100,000 | 524,288 B | 786,432 B | 1,048,560 B in 16 allocations | **0 B, 0 allocations** |
    | 50 % | 500,000 | 2,097,152 B | 3,145,728 B | 4,194,288 B in 18 allocations | **0 B, 0 allocations** |

    (The "after" figure is for the whole 1,000,000-lookup forward scan, not just
    construction. The on-disk region the cursor reads through is 20 KB / 130 KB /
    135 KB respectively, and is already mapped.)

    And it is **faster**, not a memory-for-time trade — which was the open
    question, since the `Vec` bought O(log n) random access.
    `doc_values/sparse_numeric_reader` in `benches/hot_paths.rs`, construction
    included in every arm:

    | arm | before | after | |
    |---|---|---|---|
    | `forward/n1000` (walk every present doc) | 8.168 µs | ~6.7 µs | 1.2x |
    | `forward/n10000` | 136.7 µs | ~70 µs | **~1.95x** |
    | `forward/n100000` | 1.832 ms | ~690 µs | **~2.65x** |
    | `single/n1000` (open, ask one question) | 786.8 ns | ~280 ns | ~2.8x |
    | `single/n10000` | 26.46 µs | ~279 ns | **~95x** |
    | `single/n100000` | 254.9 µs | ~213 ns | **~1200x** |

    The `single` column is where an O(cardinality) constructor hurts most and is
    the shape `lucene-search` actually has (open a reader, resolve a handful of
    docs); the `forward` column is a sort or facet count. Both improved, so
    there is no trade to weigh.
    Tests: `numeric_reader_sparse_agrees_with_numeric_value_in_any_order` — one
    field carrying all three block shapes, checked doc-for-doc against
    `numeric_value` over all 196,608 doc ids ascending, then over 3,000
    pseudo-random probes (each asked twice) — and
    `numeric_reader_sparse_region_outside_the_data_is_an_error`.

11. **[INTENTIONAL] `NumericReader::value` accepts any doc order; Java's
    `SparseNumericDocValues` does not.** Java's is a `DocValuesIterator`:
    backwards is an `assert` away from undefined. This port's doc-values API is
    random-access, so the reader compares against `DisiCursor::doc_id()` and
    calls `reset()` when a caller goes backwards. Ascending order — every real
    scan — costs one walk of the region for the whole scan; a backward step
    costs a re-walk of the block headers and still allocates nothing. Strictly
    more capable than Java at no cost to the ordered path.

12. **[PERF, not fixed — other crate] `lucene_search::field_norms::FieldNorms`
    still holds the same `Vec<i32>`.** It calls `indexed_disi::decode_doc_ids`
    at construction for exactly the reason `NumericReader` used to: it needs
    random access. The same fix applies verbatim (hold a `DisiCursor`, `reset()`
    on a backward doc), and the memory numbers in #10 are the same numbers.
    `field_norms.rs` belongs to b13, which was in flight; **recorded, not
    touched.** `lucene_search::soft_deletes` also calls `decode_doc_ids`, but
    legitimately: it builds an owned deleted-doc set, which is a different
    shape, not this one.

### Verdict

Swept (the `IndexedDISI`-composing surfaces); #10 fixed with tests and
measurements, #11 intentional, #12 handed to b13.

---

## `crates/lucene-codecs/src/norms.rs`

Java counterparts: `codecs/lucene90/Lucene90NormsProducer` (`getNorms`,
`SparseNormsIterator`), `Lucene90NormsConsumer.addNormsField`.

| Rust | Java | Verdict |
|---|---|---|
| `norm_value` | `getNorms` + `SparseNormsIterator.advanceExact` | identical, cursor-backed already; doc comment was stale (#13) |
| `write_fields`' sparse branch | `addNormsField`'s `numDocsWithValue != maxDoc` branch | rank table was missing → **fixed** (indexed_disi #4) |

13. **[No class — stale documentation, fixed]** `norm_value` carried *two*
    contradictory comment blocks: the pre-cursor one ("decodes the whole
    `IndexedDISI` region on every call … 324 µs at 100,000") immediately
    followed by the cursor one that replaced it. Both were left in place by an
    earlier edit. Replaced with one accurate note. The same stale claim in
    `docs/parity.md`'s `IndexedDISI` row (which still described the module as a
    one-shot decode with no cursor at all) was rewritten.

**No reader in this file has the #10 shape** — `norms.rs` exposes no reader
struct at all, only the `norm_value` free function, which opens a fresh cursor
per call. Checked as the task asked: `binary_value`, `sorted_ord`,
`sorted_numeric_values` in `doc_values.rs` are likewise free functions holding
no doc-id array. `NumericReader` was the only reader in either file with the
O(cardinality) `Vec`, and `lucene_search::field_norms::FieldNorms` (#12) is the
only other one in the workspace.

---

## Carry-over item 3 from `b2-packed`: the `dense_rank_power: 0` literals

`0` is not a legal value for that metadata byte (Java's domain is `-1`, stored as
`0xFF`, or `7..=15`), so these were invalid test data that happened never to
reach a DENSE block.

- `crates/lucene-codecs/src/doc_values.rs` — two literals in the BINARY test
  entry builders. **Fixed**, now `indexed_disi::NO_RANK` (newly `pub`, since
  three crates were each defining their own `0xFF`).
- `crates/lucene-codecs/src/norms.rs` — one literal in `EntryBuilder::dense`.
  **Fixed**, same way.
- `crates/lucene-search/src/field_norms.rs:556` — **left alone, correctly.**
  Checked as instructed: b13 already handled it, and its `0` is now deliberate —
  the test is `an_illegal_dense_rank_power_is_rejected_rather_than_guessed`, and
  the literal carries the comment `// not 7..=15, not 0xFF`.
- `crates/lucene-search/src/explain.rs:1687` — **recorded, not touched.** One
  more invalid literal, in a `synthetic_norms` helper whose entry is dense
  (`docs_with_field_offset: -1`), so nothing reads it. `lucene-search` was held
  by b13/b15 during this batch. One-line fix for whoever owns the file next.

---

## Gates

- `cargo fmt --all` — clean.
- `cargo clippy -p lucene-codecs --all-targets -- -D warnings` — clean for every
  file this batch touched. The command as a whole was red throughout on files
  owned by concurrent batches (`benches/blocktree_open.rs` and `blocktree.rs`,
  batch `c1-lazy-blocktree`); re-run until those cleared, and left untouched
  meanwhile.
- `cargo test -p lucene-codecs` — all pass.
- `scripts/verify-write-path.sh` — 14/14 against real Lucene 10.5.0, including
  the newly rank-tabled sparse doc-values and norms segments and the three new
  strided passes.
- `docs/parity.md` — the `IndexedDISI` row rewritten (it still described the
  module as a one-shot decode with no cursor).

### Benchmark caveat

Every number above was taken on a machine running four other sweep batches'
compilations concurrently. Run-to-run spread on a single arm reached ±10 % for
the sub-microsecond arms; the differences claimed as wins (~56x, ~450x, ~26x, ~95x,
~1200x) are orders of magnitude clear of that, and the one regression claimed
(#9, ~8 %) is inside it and is reported as approximate for that reason.
