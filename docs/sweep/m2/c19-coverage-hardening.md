# c19 — coverage hardening: the last two files under the bar, and a mechanical gate for the panic/abort class

Follow-up batch. Three items, all carry-overs from c15:

1. `check_index.rs` (89.19%) and `checksum_verify.rs` (93.00%) were the last
   two files in the workspace below the `test-coverage` skill's ≥95%-per-file
   bar — and in a 4 000-line verifier the uncovered lines are exactly where a
   check silently does nothing.
2. Nothing mechanical catches "arithmetic on a value read off disk", the
   defect class this sweep has now found by hand nine times, and which c15's
   own deliberate audit still missed eight instances of until its Tier-2
   review caught them.
3. Five negative controls in `check_index.rs` asserted `caught > 0` — an
   assertion that some corruption was caught, which passes even if almost none
   is.

Java read from **`/home/tuong/work/lucene-10.5.0`**, the pinned tag.

| Rust file | Java counterpart (10.5.0) |
|---|---|
| `crates/lucene-index/src/check_index.rs` | `lucene/core/src/java/org/apache/lucene/index/CheckIndex.java` |
| `crates/lucene-index/src/checksum_verify.rs` | `CheckIndex`'s checksum-only path + `CodecUtil.checksumEntireFile` |
| `crates/lucene-codecs/src/points.rs` (F1, F2) | `lucene/core/src/java/org/apache/lucene/util/bkd/{BKDConfig,BKDReader}.java` |
| `crates/lucene-codecs/src/norms.rs` (F3) | `lucene/core/src/java/org/apache/lucene/codecs/lucene90/Lucene90NormsProducer.java` |
| `crates/lucene-store/src/{codec_util,data_input,data_output,index_output}.rs` | `CodecUtil`, `DataInput`, `DataOutput`, `IndexOutput` |
| lint configuration | *(no Java counterpart — this is a Rust-only hazard: Java's `int` wraps where Rust's panics, and Java's `OutOfMemoryError` is catchable where Rust's allocation failure aborts)* |

Findings: **5 CORRECTNESS** (all fixed, all with tests), **2 MISSING** (both
fixed), **4 INTENTIONAL**, plus two documentation corrections, the gate itself
and the coverage work. Three of the CORRECTNESS findings and one MISSING came
out of running the new gate; one more CORRECTNESS finding and one MISSING came
out of the Tier-2 review of this batch's own work.

---

## Item 2 — the arithmetic gate

### The assessment c15 asked for

`clippy::arithmetic_side_effects` is the right lint and a workspace-wide deny
is the wrong way to turn it on. Measured across the workspace at the start of
this batch (`cargo clippy --workspace --all-targets -- -W <lint>`, unique
sites):

| lint | sites | outside `mod tests` |
|---|---|---|
| `clippy::arithmetic_side_effects` | **2 063** | 1 859 |
| `clippy::indexing_slicing` (+ slicing) | 2 371 + 371 | — |
| `clippy::cast_possible_truncation` | 1 242 | — |
| `clippy::cast_sign_loss` | 1 036 | — |

Per crate, non-test: `lucene-codecs` 1 278, `lucene-search` 237, `lucene-index`
169, `lucene-analysis` 100, `lucene-store` 33, `lucene-util` 23, `lucene-ffi`
19. A deny at that scale needs ~2 000 `#[allow]`s, which is precisely the
failure mode the brief warned about. The full assessment of each candidate,
including the ones rejected, is in [`docs/arithmetic-gate.md`](../../arithmetic-gate.md).

### The gate, and why it is not bypassable by accident

`clippy::arithmetic_side_effects` is denied through `[workspace.lints.clippy]`
in the root `Cargo.toml` and opted into **per crate** with `[lints] workspace =
true`. It runs inside `cargo clippy --workspace --all-targets -- -D warnings`
— the existing gate, the existing CI job, no new command.

The deny is **crate-wide**, so a *new* module in a gated crate is gated from
its first line and nobody has to remember to opt it in. Pre-existing modules
carry a one-line opt-out on their declaration in the crate's `lib.rs`:

```rust
#[allow(clippy::arithmetic_side_effects)] // TODO(arith-audit)
pub mod blocktree;
```

Three properties matter here:

- **Default-deny.** The only way to *avoid* the gate is to write a visible,
  reviewable `TODO(arith-audit)` line. Forgetting to do anything gets you the
  gate.
- **One burn-down list, not sixty scattered headers.** `grep -c
  'TODO(arith-audit)' crates/*/src/lib.rs` is the remaining work: 26 modules
  in `lucene-codecs`, 11 in `lucene-index`, 0 in `lucene-store`.
- **No edits to files other batches own.** Putting the markers on the `mod`
  declarations rather than at the top of each module file means adopting the
  gate touched zero of c17's or c20's source files.

Scope: `lucene-store`, `lucene-codecs`, `lucene-index` — the three crates that
turn *bytes on disk* into values. `lucene-search`/`lucene-ffi`/`lucene-core`
consume values those crates already validated; `lucene-util`/`lucene-analysis`
do not read files. Recorded as a deliberate scope, not an oversight.

### The `#[allow]` convention

Every operator must be `checked_*` with a reported corruption, `saturating_*`/
`wrapping_*` where that is the honest semantics, or a plain operator under an
`#[allow]` carrying an `// ARITH:` comment that **names the invariant**. An
`#[allow]` without an `// ARITH:` proof is a review failure. Tests, benches,
examples and fixture builders opt out at their own file/module boundary, with
the reason stated. All of it is in `docs/arithmetic-gate.md`, which
`AGENTS.md` and the `code-review` skill now point at.

### Fully audited under the gate

- **`lucene-store`** (whole crate): 37 sites. Every one turned out to be
  provably safe — this crate's cursor arithmetic is bounded by its own
  `pos <= buf.len()` invariant and by explicit guards. Resolution: 8 `// ARITH:`
  justified allows (the vint/vlong/zlong shift accumulators, `read_group_vints`,
  the whole `impl DataInput for SliceInput`, `header_length`/
  `index_header_length`, `write_group_vints`), two guard-and-subtraction merges
  that make the invariant local (F7), and one `saturating_add`.
- **`lucene-index/src/check_index.rs`** and **`checksum_verify.rs`** (this
  batch's own files): 48 sites, 3 of them real (F3, F5, F8 below), the rest counters
  now `saturating_add`. Both files carry `#![deny(clippy::arithmetic_side_effects)]`
  in their own right, with the reason stated: **a verifier that panics on a
  corrupt file has failed at its one job.**

### What running it across the workspace found

The lint reports 2 063 sites; the ones that are *defects* are the ones where an
operand comes off disk. In the audited scopes that is five, all fixed, all
with a test that fails without the fix. Outside them, the burn-down markers
record the rest as unaudited rather than as clean — a claim of "nothing else
found" would be false, because 1 447 sites in `lucene-codecs`/`lucene-index`
have not been looked at yet. That is the honest state and it is what the
markers say.

### F1 `[CORRECTNESS → fixed]` `.kdm`'s `numDims x bytesPerDim` overflowed an `i32`

**Java** (`BKDConfig`'s canonical constructor, 10.5.0): bounds `numDims`
(1..=16) and `numIndexDims` (1..=8) but puts **no upper bound at all** on
`bytesPerDim` — only `bytesPerDim > 0`. `BKDReader` then does `new
byte[config.packedIndexBytesLength()]` where that method is
`numIndexDims * bytesPerDim`. The product wraps an `int`, goes negative, and
`new byte[negative]` throws `NegativeArraySizeException` — a caught,
reportable corruption.

**We did**: ported the same bounds, so `check_config` accepted
`bytesPerDim = 2^30`, and `(num_index_dims * bytes_per_dim) as usize`
**panicked** with `attempt to multiply with overflow` in a debug build. A
panic is not what the caller is prepared for, and through the FFI in a debug
build it is a dead JVM.

**Fixed**: `check_config` now rejects a `numDims x bytesPerDim` that overflows,
reproducing Java's *outcome* without Java's mechanism. Test:
`num_dims_times_bytes_per_dim_overflowing_is_rejected` (a five-byte vint for
`bytesPerDim` spliced into a real writer-produced `.kdm`; it panics without
the fix).

### F2 `[CORRECTNESS → fixed]` Two `.kdm`-sized allocations, one step further out

The product that does *not* overflow is worse than the one that does.
`numIndexDims = 1, bytesPerDim = 2^30` is a legal-looking `.kdm` that asks
`vec![0u8; n]` for two 1 GB buffers out of a few hundred bytes of file. In
Java that is an `OutOfMemoryError` the caller can catch; in Rust an allocation
failure is an **abort**, which no `catch_unwind` at the FFI boundary can
intercept.

**Fixed**: the bytes have to be in the `.kdm` for the subsequent `read_bytes`
to succeed anyway, so requiring `meta_input.remaining() >= 2 * length` up
front costs nothing and turns the abort into a decode error. Test:
`a_packed_value_length_larger_than_the_kdm_is_rejected_before_allocating`.

### F3 `[CORRECTNESS → fixed]` `.nvm`'s `docsWithFieldOffset + docsWithFieldLength` overflowed an `i64`

**Java**: `Lucene90NormsProducer` calls `data.slice(entry.docsWithFieldOffset,
entry.docsWithFieldLength)`, which takes the two **separately** and
range-checks each against the file. There is no sum to overflow.

**We did**: `norms::norm_value` built the range as
`offset as usize .. (offset + length) as usize`. Both halves are `i64`s read
straight off `.nvm` with no relationship between them on the wire, so a
corrupt file can pick any pair: `offset = i64::MAX, length = 1` panics with
`attempt to add with overflow` before `data.get` ever sees a range. A negative
offset separately sign-extended into a ~2^64 slice start.

**Fixed**: `norms::sparse_region`, `checked_add` + `usize::try_from`, both
failure modes becoming the `Eof` the caller already handles. Test:
`a_sparse_entry_whose_offset_plus_length_overflows_is_rejected` (both halves).
**The identical expression was duplicated in `check_index.rs`'s
`check_field_norms`** and is fixed there in the same shape.

### F4 `[INTENTIONAL]` `check_doc_value_skipper`'s `maxDocID(0) + 1`

Reported by the gate and, on inspection during the Tier-2 review, **not** a
defect: `NO_MORE_DOCS` *is* `i32::MAX`, a fresh skipper starts at `-1`, and the
loop's `break` catches the sentinel one line before the next increment, so the
`+ 1` cannot overflow. Now `saturating_add` under rule 2 of the gate
(saturation unreachable, clippy cannot see it), with a comment that says that
rather than inventing a reachable corruption — the first version of this
finding claimed one, and would have taught the next reader a false pattern.
Recorded here rather than deleted because a reader coming from the diff will
otherwise ask why the operator changed.

### F5 `[CORRECTNESS → fixed]` `.tim`'s term statistics accumulated into a panicking `+=`

`summed_doc_freq += claimed.doc_freq as i64` and `summed_total_term_freq +=
claimed.total_term_freq` run once per term over values read straight off
`.tim`/`.tmd`. `totalTermFreq` is a `long` in Lucene, so a corrupt dictionary
can hand this loop `i64::MAX` repeatedly and overflow the accumulator — inside
the check that exists to catch exactly that dictionary. Both are now
`saturating_add`, which leaves the mismatch reportable (the re-derivation
below still disagrees) instead of panicking. Same treatment for ~20 other
counters in the file.

### F6 `[MISSING → fixed]` `checksum_verify`'s two `.si` failure arms had never run

`verify_directory` has a deliberate, documented design decision: a segment
whose `.si` cannot be opened, or cannot be parsed, is reported as a single
failed `.si` entry rather than silently skipped — because without the `.si`
there is no file list for that segment, and skipping it would report "all
passed" for an index with a whole unreadable segment. Neither arm was
exercised by any test, so the design decision was unverified.

Tests: `a_missing_si_is_reported_and_the_other_segment_still_verified` (which
also asserts the *other* segment's files are still checked, the half of the
decision that is easy to regress) and
`an_unparsable_si_is_reported_rather_than_skipping_the_segment`.

### F7 `[INTENTIONAL]` `lucene-store`'s 37 sites were all provably safe

No defect in the crate every file-derived value passes through. Two changes
are still improvements rather than no-ops: `check_footer` and
`retrieve_checksum` had a `if len < FOOTER_LENGTH { return Err }` guard
followed by a separate `len - FOOTER_LENGTH`, and both are now one
`checked_sub` — the guard and the subtraction are the same operation, and
writing them once makes it impossible for a future edit to move one without
the other.

### F8 `[CORRECTNESS → fixed]` A `.tmd`'s term statistics could saturate into agreeing with themselves

Found by the Tier-2 review, in this batch's own work. `summed_total_term_freq`
was converted to `saturating_add` along with every other counter — but unlike
the others its operand is an `i64` read off disk *and* it is compared against
`sum_total_term_freq`, another `i64` read off disk. A `.tmd` claiming
`sumTotalTermFreq = i64::MAX`, paired with per-term values that overflow, would
have compared **equal**: `postings.field_summary` would pass on exactly the file
it exists to reject.

**Fixed**: `checked_add`, with the overflow pushed as a `term-stat` problem —
which is the honest report and costs nothing on a sane file. The neighbouring
`summed_doc_freq` stays saturating and is justified in place (its operand is an
`i32`, so reaching `i64::MAX` takes 2^32 terms).

### F9 `[MISSING → fixed]` `BKDConfig`'s `maxPointsInLeafNode` ceiling was unported

Also from the review. `check_config`'s doc comment claims to reproduce
`BKDConfig`'s constructor validation; it reproduced four of the five guards.
`maxPointsInLeafNode > ArrayUtil.MAX_ARRAY_LENGTH` was missing. That value
sizes the per-leaf point buffer, so it is an allocation length read off disk —
the same class as F2, one guard away. Ported with Java's exact ceiling
(`i32::MAX - 16`, Java's `Integer.MAX_VALUE - NUM_BYTES_ARRAY_HEADER`) so this
port rejects exactly the sixteen values Java rejects. Test:
`max_points_in_leaf_node_above_the_array_ceiling_is_rejected`.

### F10 `[INTENTIONAL]` `clippy::disallowed_methods` on `Vec::with_capacity` was assessed and declined

The allocation shape is the one that *aborts*, so it is the one worth catching
most. But `clippy.toml` is workspace-global with no per-crate scoping, and 270
of the workspace's `Vec::with_capacity` call sites are sized by this port's own
in-memory data. Recorded in `docs/arithmetic-gate.md` with the reason; the
audited modules cap every disk-sized reservation by hand instead.

---

## Item 3 — the weak-floor negative controls

c15's shape, applied to all five `caught > 0` assertions in `check_index.rs`:
**re-sign the footer over every corruption** so `file:*`'s CRC cannot "catch"
it and only semantic checks can fire; **assert a specific count** with the
measured number in the failure message; and **drive one case through
`check_directory`** asserting it trips the intended check *and nothing else*.

Re-signing is the part that makes the numbers mean anything. Without it every
byte flip is trivially "caught" by the checksum, `!failed.is_empty()` is
vacuous, and the only real claim left is `caught > 0`.

How many corruptions each check actually rejects, measured:

| control | corruptions | rejected by the named check | by another check | accepted | isolated case? |
|---|---|---|---|---|---|
| `.nvm` → `norms.*` | 99 | **85** | 0 | 14 | yes |
| `.tip` → `postings.seek_agrees` | 99 | **44** | 12 | 43 | yes |
| `.vex` → `hnsw.neighbors_*` | 318 | **138** | 3 | 177 | yes (315 got past `hnsw.open`) |
| `.dvd` → `doc_values.terms_sorted`/`ords_dense` | 99 | **18** | 27 | 54 | yes |
| `.dvd` (numeric + binary) → `doc_values.*` | 261 | **69** | 0 | 192 | yes |
| `.doc` → `postings.advance_agrees` | 2 034 | **5** | 2 026 | **3** | **no** |

Each row now asserts a floor at roughly 85% of the measured number, with the
measured number in the message, plus an isolation assertion.

(The numeric/binary `.dvd` row is a new control this batch added, not a
rewritten one — `doc_values_index` carries dense numeric, GCD-compressed
numeric, sparse numeric, fixed-length binary, variable-length binary and sparse
binary entries, six decode paths that only the SORTED fixture's control had
ever touched.)

Four of these numbers are findings in their own right:

- **`.nvm` and `.dvd`: 0 caught by another check.** Nothing else in the module
  reads either file, so a corruption the norms / doc-values checks miss is a
  corruption nothing catches. Both now assert `caught_by_other == 0`, which
  turns the floor from a nice-to-have into the file's only line of defence.
- **`.doc`: no isolated case exists.** None of 2 034 corruptions tripped
  `postings.advance_agrees` alone: a `.doc` byte that makes the skip list
  disagree with the decoded blocks almost always also makes a doc ID
  non-increasing, which `postings.doc_ids_valid` sees first. The check is a
  *second* witness, not a sole one — still worth having (it is the only place
  the skip data and the blocks it indexes are compared) but not falsifiable in
  isolation by byte corruption. The test now pins that fact with a message
  telling the next reader what to do if it ever changes, rather than leaving
  an isolation assertion that cannot hold. The number that *is* worth
  asserting there is the other one: with the CRC out of the picture, all but
  **3** of 2 034 `.doc` corruptions are rejected on their semantics alone.
- **`.tip`: 43 of 99 accepted, and that is correct.** The `.tip` trie is an
  *index* into the `.tim`; many of its bytes only change which block a seek
  starts scanning from, and a scan that starts in the right block still finds
  the right term. A control that demanded 99/99 would be demanding a wrong
  answer.

The `.vex` control also changed shape: it used to stop as soon as it had three
hits, so the count it asserted could never say how far the checks reach. It
now runs a fixed uniform sample (every 2 111th byte x three masks) and reports
over the whole sample.

---

## Item 1 — the two files under the bar

### What the uncovered paths were

This is the part that matters more than the number. `check_index.rs`'s 510
uncovered lines were **not** spread evenly: they were almost entirely
`problems.push(..)` / `Check::fail(..)` arms. Roughly 150 individual checks in
a 4 000-line verifier had never once been observed to fire. A check that
silently does nothing is indistinguishable, from the outside, from a check
that passes — which is the exact failure mode this module exists to prevent,
turned on itself.

The largest blocks, and what they were:

| block | lines | what had never fired |
|---|---|---|
| `check_one_vector_field`'s self-consistency arms | ~85 | **all 13** of `checkFields(.., isVectors = true)`'s invariants: term order, `freq <= 0`, positions-vs-freq, negative/backwards positions, missing positions on a positions field, offset counts, negative/inverted offsets, missing offsets, half-present offsets, stray payloads |
| `check_one_vector_field`'s postings cross-check | ~50 | **all 7** arms of `testTermVectors`' slow level — the vector disagreeing with the inverted index on existence, freq, positions, offsets or payloads |
| `check_doc_values`' `entry_present` arms | ~25 | all five: a `.fnm` claiming a doc-values type the `.dvm` has no entry for |
| `check_doc_value_skipper`'s global guards | ~12 | all four of `checkDocValueSkipper`'s pre-walk guards |
| `checksum_verify::verify_directory`'s `.si` arms | ~14 | both (F6) |
| `SortedSetKind::Single` in `check_doc_values` | ~24 | a whole *format shape* — a single-valued SORTED_SET field — that no fixture carries |

### What was done about it

Four new tests, all driven at the function rather than by corrupting bytes,
which is what makes them cheap enough to cover an arm apiece:

- `every_term_vector_self_consistency_arm_reports_its_own_invariant` — a
  hand-built `TermVectorField` per invariant, asserting the arm fires **and
  that no other arm does** (`assert_eq!(problems.len(), 1)` throughout), which
  is what proves the arms are distinguishable from one another.
- `every_term_vector_postings_cross_check_arm_reports_its_own_disagreement` —
  hand-built vectors against real hand-written positional postings. There is no
  term-vector *writer* in this port that can be told to emit a wrong freq, and
  a byte flip in a real `.tvd` fails the decode long before it produces a
  well-formed vector that merely disagrees, so the function is the only
  reachable seam.
- `the_doc_values_skipper_global_guards_each_fire_on_their_own_input` — a
  hand-built `DocValuesSkipIndex`. The `.dvs` corruption sweep c15 added cannot
  reach these four: they read `.dvm`-derived summary fields the sweep does not
  touch.
- `a_fnm_claiming_a_doc_values_type_the_dvm_lacks_is_caught_for_every_type` —
  and this one is a real scenario, not a contrivance: `Lucene90DocValuesProducer`
  routes each `.dvm` entry by the type byte in the **`.dvm`**, while every
  reader asks for a field's values by the type in the **`.fnm`**. When the two
  disagree the entry becomes unreachable and the field silently reads as "no
  doc values at all" — every range query, sort and facet on it quietly returns
  nothing, while the `.dvd` still decodes perfectly and every other check
  passes.

- `a_fnm_claiming_norms_the_nvm_does_not_have_is_caught` and
  `a_fnm_disagreeing_with_the_vemf_about_a_vector_field_is_caught` — the same
  cross-file shape for norms and vectors. The vector one is the sharpest: a
  `.fnm`/`.vemf` disagreement about a field's *similarity function* is
  completely silent, because the distances still compute, they are just the
  wrong distances.
- `a_segment_that_disagrees_with_its_own_term_vectors_is_caught` — a `.si`
  listing only some of `.tvd`/`.tvx`/`.tvm`, a `.si` `docCount` above the
  term-vectors reader's own `maxDoc`, and a `.fnm` denying vectors a document
  actually carries.
- `a_file_replaced_by_another_kind_is_reported_by_its_own_open_check` — sixteen
  cases, one per `(fixture, victim file, donor file)`: every codec file the
  `.si` lists, overwritten by a well-formed file of a *different* kind from the
  same segment. This is the failure `check_index` most needs to survive
  gracefully, and it asserts the module's actual contract: report under the
  right subsystem's name, keep checking the rest of the segment, do not panic.
  Every `Err(e) => checks.push(Check::fail("<x>.open", ..))` arm in the file
  was unexercised before it.
- `a_single_valued_sorted_set_field_is_checked_through_the_single_branch` —
  reaches `SortedSetKind::Single`, a whole second SORTED_SET decode path with
  its own ordinal bookkeeping that no fixture in the repo exercised, because
  every Java-written SORTED_SET fixture here happens to be multi-valued.
- `an_index_sort_on_a_multi_valued_sorted_numeric_field_is_verified` — the
  SORTED_NUMERIC arm of `sort_key_values`, including the `min()` that makes
  sorting on a multi-valued field well-defined (Lucene's `MIN` selector). The
  negative case sorts by each document's *maximum* instead, which is the
  mistake a writer that picked the wrong selector would make.
- `soft_deletes_are_counted_only_among_live_documents` — `checkSoftDeletes`'
  `del_gen != -1` branch, which intersects the soft-deletes field with the
  `.liv` so a document that is both soft- and hard-deleted is not counted
  twice. Every soft-deletes fixture here had `del_gen == -1`, so the check had
  only ever run with `live = None`.

Plus F6's two `checksum_verify` tests, and the five rewritten negative controls
from Item 3 (which reach a further set of decode-error arms).

### Coverage

| file | before | after | bar |
|---|---|---|---|
| `crates/lucene-index/src/checksum_verify.rs` | 93.00% | **97.11%** | ✅ |
| `crates/lucene-index/src/check_index.rs` | 89.19% | **94.19%** | ❌ 0.81 points short |

(`cargo llvm-cov -p lucene-index`, run into a private `CARGO_TARGET_DIR`. That
last part is not incidental: `cargo llvm-cov` merges every `*.profraw` under
`target/llvm-cov-target`, so a concurrent batch running coverage in the same
worktree silently poisons the result. A run mid-batch reported `check_index.rs`
at 61% with 7 791 instrumented lines and a 32% workspace total; the same tree
measured in an isolated target directory reported 93%. Worth knowing before
anyone chases a phantom regression.)

`checksum_verify.rs` is over the bar. **`check_index.rs` is not**, and that is
this batch's one unmet requirement — it moved 89.19% → 94.19%, and the residual
is not a block that one more test closes.

What is left, precisely: **139 uncovered region starts in production code**
(~290 lines), and after this batch's work the largest remaining block is
**11 lines**; the next five are 9, 9, 6, 6 and 5. It is ~110 individual
`problems.push(..)` / `Check::fail(..)` arms, each needing its own contrived
input, spread across `check_postings`' term-statistic and
seek/intersect/advance disagreements, the per-subsystem per-ordinal
decode-error arms, and the commit-level `segments_N`-vs-`.si` version guards.
The remaining **40 uncovered lines in the test module** are
`assert!(cond, "…{:?}", x)` message closures, which by construction only
evaluate when a test fails; they are ~0.7 points of permanently unreachable
denominator.

Estimated to close: 4–8 more tests of the same kind as the twelve added here
(47 more covered lines takes it over the bar).
Recorded as a carry-over rather than half-done, because the *valuable* half —
naming the 150 checks that had never been observed to fire, and firing the
largest families of them — is done, and the rest is a long tail of one-arm-
per-test work that another batch can pick up from the list above.


---

## Gates

- `cargo fmt --all` — clean.
- `cargo clippy -p lucene-store -p lucene-codecs --all-targets -- -D warnings` — clean.
- `cargo clippy -p lucene-index --all-targets -- -D warnings` — clean for this
  batch's files; the only remaining diagnostics are `clippy::type_complexity`
  warnings in `index_writer.rs`, from c17's in-flight work. See Handoffs.
- `cargo clippy --workspace --all-targets -- -D warnings` — blocked at the time
  of writing by a compile error in `lucene-ffi/src/writer.rs` (a non-exhaustive
  match on `index_writer::Error`, from c16/c17's in-flight changes). Nothing in
  this batch's scope contributes to it; every crate this batch touched is clean.
- `cargo test -p lucene-store -p lucene-codecs` — **all green** (37 test
  binaries).
- `cargo test -p lucene-index --lib -- check_index checksum_verify` — **85
  passed, 0 failed**, including
  `every_real_lucene_index_fixture_passes_every_check`.
- `python3 scripts/check-arith-allows.py` — ok (37 modules still unaudited),
  the new check this batch adds to the pre-commit gate.
- `cargo test -p lucene-index` (whole crate) — 7 failures, **all in
  `merge::tests`**, from another batch's in-flight refactor of `merge.rs`
  (`MergedDocValuesField`, `build_doc_id_maps`' new arity). Not this batch's
  files; the crate did not compile at all for part of this batch for the same
  reason.
- `scripts/verify-write-path.sh` — **20/20** confirmed (not assumed) mid-batch,
  and **21/21** at the end, a concurrent batch having added
  `VerifySortedSegment <- write_sorted_merged_segment_fixture` while this one
  ran.
- `python3 scripts/check-parity.py` — ok.

### Runtime cost

**Of the verifier: none measurable.** Everything added to production code is
either a `saturating_*` in place of a `+=` (one `cmov`, on counters that
already ran once per document/term), one `checked_add` + `try_from` per norms
*field* at open, or one `checked_mul` and one length comparison per points
*field* at open. No new pass over any document, term, ordinal or byte.
`every_real_lucene_index_fixture_passes_every_check` — `check_directory` over
every Java-written fixture in the repo — runs in **0.28 s** in release, stable
across runs.

**Of the tests**, which is where the cost actually landed:

| | wall clock |
|---|---|
| the five rewritten negative controls | 55 s total (`.tip`+`.nvm`+`.vex`+`.dvd` 54 s, `.doc` 0.5 s) |
| the twelve new tests | ~8 s total (the file-replacement one is 5 s: sixteen `check_directory` runs over four fixtures) |
| `check_index` + `checksum_verify` suite | ~50 s (87 tests) |

The `.vex` control is most of the 55 s and is the one that got *cheaper* in
kind: it used to stop at three hits, which made it fast but uninformative.
Re-signing itself is free — a CRC over a file this port already reads whole.


---

## Tier-2 review (`quality-reviewer`)

Run on this batch's scope with the concurrently-edited files excluded. It read
the Java alongside and independently re-verified `BKDConfig`'s guard list, the
`vectors.dimension_positive` withdrawal, `norms::sparse_region`, and every
`// ARITH:` proof in `lucene-store` and `check_index.rs`.

**Two gating findings, both about a proof or a claim being *wrong* rather than
missing** — which is the failure mode this gate introduces and the reason the
mechanical check below cannot replace review:

1. **The `checkDocValueSkipper` ARITH comment described a corruption that
   cannot happen.** It claimed a corrupt `.dvs` could put `i32::MAX` in
   `maxDocID(0)` "without it being `NO_MORE_DOCS`'s sentinel run". But
   `NO_MORE_DOCS` *is* `i32::MAX`, and the loop breaks on it one line before
   the next increment — so the `+ 1` could never overflow. `saturating_add` is
   still right (rule 2: saturation unreachable, clippy cannot see it), but the
   comment now says that instead of inventing a reachable case. The next reader
   would otherwise have copied the pattern to somewhere it *is* reachable and
   believed it handled.
2. **`docs/parity.md` claimed 10.5.0's `readVInt` is bounded.** It is not:
   `for (int shift = 7; (b & 0x80) != 0; shift += 7)`, unbounded. This port's
   5-byte cap is a deliberate divergence, and `data_input.rs`'s own module doc
   says so — parity.md said the opposite of the file it documents. Corrected.
   (Pre-existing, from b1; found because the reviewer checked the claim against
   the Java rather than against the diff.)

**Nine advisories, eight acted on:**

- `summed_total_term_freq` was `saturating_add` over an `i64` read off disk,
  compared against another `i64` read off disk — so a `.tmd` claiming
  `i64::MAX` paired with overflowing per-term values would compare **equal**
  and the check would pass on exactly the file it exists to reject. This was
  the one saturating conversion in the batch that violated the batch's own
  rule. Now `checked_add` with a reported `totalTermFreq values overflow i64`
  problem. Every other saturating counter was traced and is bounded by
  `si.doc_count` or compared against an `i32`-derived value.
- Two `// ARITH:` proofs in `lucene-store` had correct conclusions and wrong
  reasoning: `header_length`'s named `write_header` as *rejecting* a long codec
  name when it only `debug_assert!`s (the real bound is `str::len() <=
  isize::MAX`), and `write_group_vints`' stated range `0..=4` for
  `leading_zeros() / 8` would have made `lens[k] - 1` underflow — the actual
  invariant is the `| 1`, which is now what the comment names.
- The impl-block-scope allow on `impl DataInput for SliceInput` is wider than
  the convention prefers; it now says explicitly that a method added there must
  re-verify the invariant.
- The single-valued SORTED_SET control counted "any check failed" rather than
  the ordinal-space checks it was named for — the exact weakness Item 3 exists
  to remove, reintroduced in a new test. Now attributed by check name.
- All seven corruption sweeps had `Err(_) => continue`, which silently shrank
  the denominator the floors are stated against. An unreadable commit is now
  counted as caught by another check, so `total` equals the sweep's iteration
  count.
- `check_config` was missing `BKDConfig`'s `maxPointsInLeafNode >
  ArrayUtil.MAX_ARRAY_LENGTH` guard while its doc comment claimed to enumerate
  them all. Added with Java's exact ceiling (`i32::MAX - 16`) and a test.
- `docs/sweep/m2/c9-check-index.md` still listed `vectors.dimension_positive`
  as shipped; corrected inline, the way c18's corrections were.
- The `assert!(isolated.is_none())` tripwire in the `.doc` control was flagged
  as failing-on-an-improvement; kept deliberately, since the message tells the
  next reader exactly what to do.

**The reviewer's suggested mechanical check is implemented**:
`scripts/check-arith-allows.py`, wired into `.githooks/pre-commit`. It requires
every `#[allow(clippy::arithmetic_side_effects)]` under `crates/*/src/` to be
either a `TODO(arith-audit)` marker or preceded by an `ARITH:` comment block,
rejects a module-scope `#![allow]` outside a `#[cfg(test)]` module, and checks
the burn-down counts in `docs/arithmetic-gate.md` against the markers actually
in the tree. **It found a real unjustified `#[allow]` on its first run** (the
second one in `open_postings_bytes`, whose `ARITH:` block covered only the
first). What it cannot check is whether a proof is *true* — findings 1, 4 and 5
above are all proofs that existed and were wrong, and only a reader catches
those.

**Verified clean by the review, no action**: the `vectors.dimension_positive`
withdrawal (checked against `CheckIndex.java:2798-2808`, `FieldInfo.java:653`
and `:225`, and this port's `field_infos.rs:342`); `norms::sparse_region` and
its sharing between the read path and `check_index`; all five `ARITH:` proofs
in `check_index.rs`, including the points one, re-derived against
`read_leaf_block`'s own sizing; the burn-down counts; dependency direction;
`#![forbid(unsafe_code)]` intact.


## Handoffs

Recorded rather than fixed, because the files belong to in-flight batches:

- **c17 — `crates/lucene-index/src/index_writer.rs`**: three
  `clippy::type_complexity` warnings (626, 1204, 3250). Fail `-D warnings`.
  (An earlier `clippy::clone_on_copy` at 4976 was fixed while this batch ran.)
- **c16/c17 — `crates/lucene-ffi/src/writer.rs:222`**: non-exhaustive match on
  `index_writer::Error` (five new variants). Blocks the workspace clippy gate.
- **`crates/lucene-index/src/merge.rs`**: seven `merge::tests` failures from an
  in-flight refactor.

Edits this batch made to files it does not own, all mechanical and minimal:

- **`crates/lucene-codecs/tests/postings_skip_pointers.rs`** (c20's new
  integration test, created while this batch ran): a five-line
  `#![allow(clippy::arithmetic_side_effects)]` opt-out header, the documented
  convention for test-support files. Without it the crate-wide deny broke
  c20's build.
- **63 other `tests/`/`benches/`/`examples/` files** across `lucene-store`,
  `lucene-codecs` and `lucene-index`: the same header.
- **`crates/lucene-codecs/src/{points,norms}.rs`**: F1–F3. Neither is owned by
  an in-flight batch.

## Carry-over

- [ ] **`check_index.rs` is at 94.19%**, still 0.81 points under the bar. The
      work list is the 139 uncovered production region starts enumerated
      above; the largest block is 11 lines, so it is ~110 individual
      never-fired check arms — 47 covered lines takes it over the bar, roughly
      4–8 tests of the kind this batch added.
      Note that ~0.7 points of the gap is unreachable by construction (assert
      message closures in the test module).
- [ ] **The arithmetic gate's burn-down**: 26 modules in `lucene-codecs` and 11
      in `lucene-index` still carry `TODO(arith-audit)`. `points.rs` (183
      sites), `for_util.rs` (166), `fst.rs` (105), `blocktree.rs` (103) and
      `term_vectors.rs` (99) are the largest. `postings.rs` is the highest
      *value* — five defects of this class have come out of it — and c15
      already hardened it, so its audit should be short; it was left because
      c20 owns it.
- [ ] **`lucene-search`, `lucene-ffi`, `lucene-analysis`, `lucene-util` are not
      gated.** Deliberate (they consume already-validated values, or read no
      files) but worth revisiting for `lucene-util`, whose bit-set and SIMD
      code does index by values that ultimately came off disk.
- [ ] **`clippy::indexing_slicing`** covers the half of the class
      `arithmetic_side_effects` does not (b4's block region lengths, b6's
      live-docs ghost bits). Too noisy to switch on globally; worth adopting
      per module *during* each module's arithmetic audit, while the code is
      already open.
- [ ] **A single-valued SORTED_SET fixture from real Java.** This batch reaches
      `SortedSetKind::Single` with this port's own writer, which is the right
      test for `check_index` but not a differential one for the *format*. Every
      Java-written SORTED_SET fixture in the repo happens to be multi-valued,
      so the collapse rule itself has never been checked against Lucene's.

