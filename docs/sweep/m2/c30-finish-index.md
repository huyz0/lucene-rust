# c30 — finishing `lucene-index`: the last check-index arms, and the last three arithmetic-gate modules

The final batch of the M2 sweep. It closes the two items left open in
`lucene-index`:

1. **`check_index.rs`'s residual failure arms.** c25 took the file to 97.25%
   and accounted for its arms exactly — 45 driven, 10 deleted as unreachable,
   ~39 remaining, each listed with the fixture shape that would close it.
2. **The last three `TODO(arith-audit)` markers in the workspace** —
   `index_writer`, `merge`, `merge_policy`. `lucene-store` and `lucene-codecs`
   were finished by c24/c27/c31; these three were the remainder.

Java read from **`/home/tuong/work/lucene-10.5.0`**, the pinned tag.

| Rust file | Java counterpart (10.5.0) |
|---|---|
| `crates/lucene-index/src/check_index.rs` | `index/CheckIndex.java` (`checkFields`, `testPostings`, `testStoredFields`, `testTermVectors`, `testDocValues`, `testPoints`, `testSort`, `checkSoftDeletes`, `checkDocValueSkipper`, `getConnectedNodesOnLevel`) + `index/SegmentInfos.java`'s `readCommit` |
| `crates/lucene-index/src/checksum_verify.rs` | `CheckIndex`'s checksum-only path + `codecs/CodecUtil.java`'s `checksumEntireFile` |
| `crates/lucene-index/src/index_writer.rs` | `index/IndexWriter.java`, `index/IndexingChain.java`, `index/SegmentMerger.java` |
| `crates/lucene-index/src/merge.rs` | `index/SegmentMerger.java`, `index/MultiSorter.java`, `codecs/lucene90/compressing/Lucene90CompressingStoredFieldsWriter.java`'s `merge`, `codecs/PointsWriter.java`'s `merge` |
| `crates/lucene-index/src/merge_policy.rs` | `index/TieredMergePolicy.java`, `index/MergePolicy.java` |
| `crates/lucene-codecs/src/vectors.rs` (cross-batch) | `codecs/lucene99/Lucene99FlatVectorsReader.java`, `codecs/lucene95/OrdToDocDISIReaderConfiguration.java` |

**Totals: 7 `CORRECTNESS` (all fixed, all with a test that fails against the
unfixed code bar one, stated below), 1 defensive slicing fix, 1 arm deleted as
unreachable, 5 kept as unreachable *error handling* with the proof at the site,
1 pre-existing clippy warning cleared, 66 arithmetic-gate lint sites resolved,
17 new tests.** `check_index.rs` 97.25% → **98.58%**, and
`docs/arithmetic-gate.md`'s burn-down table goes **3 → none**: the whole
workspace is now audited.

Two of the `CORRECTNESS` findings are process kills rather than wrong answers:
one **verified SIGABRT** (4.29 GB, `signal: 6`) and one `FixedBitSet` index
panic. Both are in the *merge* path, which an `IndexWriter` runs on its own
without a caller asking for it.

---

## Starting state

A crashed earlier attempt at item 1 had left its code in the tree and its
report (`c30-check-index-arms.md`) full of `PLACEHOLDER` markers where the
measurements go, and — the part that mattered — had **deleted the three
`TODO(arith-audit)` markers from `crates/lucene-index/src/lib.rs` without doing
the audit**, exactly as an earlier attempt had done in `lucene-codecs`. c31
restored the markers before this batch began, which is the right call: the
marker is the documented opt-out, and removing one is the *last* step of
auditing a module, not the first.

So this batch's starting point was:

- item 1's code present and green (`cargo test -p lucene-index` — 671 lib
  tests, 0 failures), its claims unverified and its numbers unmeasured;
- item 2 not started, with `cargo clippy -p lucene-index --all-targets`
  reporting **68 `arithmetic_side_effects` diagnostics** the moment the markers
  come off (66 in lib code, 2 in a test module);
- one pre-existing `clippy::ptr_arg` error at `check_index.rs:11824`
  (`patch_kdm_doc_count(kdm: &mut Vec<u8>, ..)`), which was the *only*
  diagnostic in the whole workspace at the start of this batch.

Everything item 1's report claimed was re-checked against the tree rather than
trusted; the claims that were verifiable by inspection (`file_region` in
`vectors.rs`, `is_live`/`is_live_at`, the `MAX_POSITION` import,
`connected_nodes_on_level`'s deleted guard) are all present, and every test it
names passes.

---

# Item 1 — `check_index.rs`

## The judgement this batch adds to c25's rule

c25 established: *a shipped check must be falsifiable*, and deleted ten arms
that could not fire. Applying that rule to the residue turns up a family it
does not cover, and the distinction is worth stating because a mechanical
reading of c25 would make this file worse.

**A `Check::fail` arm that cannot fire is a false claim of coverage. An
`Err(e) =>` arm that cannot fire is total error handling.**

The first claims "this invariant is guarded"; nothing can redeem the claim, a
reader ticks it off, and it is worse than no check. The second is a `Result`
the decoder's *signature* forces this caller to handle; the only ways to
remove it are to ignore the error or to unwrap it, in a verifier whose one
contract is that a corrupt file produces a report rather than a panic.

Six arms fall on that line and are **kept, with the proof written at the site**
rather than deleted:

| arm | why it cannot fire today |
|---|---|
| `check_vectors`' `float_vector_values` / `byte_vector_values` `Err` | `flat.field()` already returned `Some` and the `match` is on *that entry's* encoding, so `entry()` cannot fail; `raw_values`' bound was proved by `read_field_entry` (`vectorDataOffset + vectorDataLength <= .vec length`, in `u128`). |
| the byte branch's per-ordinal `values.vector(ord)` `Err` | `bytes()` rejects only `ord < 0 \|\| ord >= size`, and `size` is this loop's own bound. |
| `check_hnsw_graphs`' `sorted_nodes_on_level` `Err` | the loop's bound is `graph.num_levels()` and `read_field_entry` sizes `nodes_by_level` to exactly that. |
| `check_doc_values`' `DocValuesType::None => unreachable!` | required for match exhaustiveness over a set the caller has already filtered. |
| `check_postings`' `debug_assert!(false, ..)` for a field the `FieldInfos` lacks | c25's D9 shape (`blocktree::open` takes every field *name* from this very `FieldInfos`), but written as a `debug_assert` + `continue` rather than a reported check — so it is not a false coverage claim, and the `let Some(..) else` binding needs the branch regardless. c25 left this as an open D-list question; **it is closed as a keep**, for the same reason as the four above. |

The one arm that *was* a claim, and is deleted, is D11.

## D11 `[deleted]` `connected_nodes_on_level`'s entry-point guard

c25 left this as an explicit open question: *"prove whether `read_field_entry`
can produce `size == 0` with a non-zero `vectorIndexLength`. If it cannot, the
guard joins D1–D10."*

It can — and the graph still never reaches this function, which is the answer.

- `entry_node` is `0` for a graph with `numLevels <= 1`, and
  `nodes_by_level[top][0]` otherwise, which `read_field_entry` validates into
  `0..size`. So `entry >= size` requires `size == 0`.
- A field with `numLevels <= 1` contributes `size` node offsets, so `size == 0`
  leaves `numberOfOffsets == 0`, hence no `offsetsMeta` — and
  `OffHeapHnswGraph::new` rejects that pair outright as *"graph has data but no
  node offsets"*, reported as `hnsw.open:<field>`.
- A field with `numLevels >= 2` must satisfy `0 < numNodesOnLevel <= size` on
  its upper level, so `size >= 1`.

The guard is gone; the depth-first loop below it still range-checks every node
it pops, which is what keeps the walk total. The proof is left at the site.
**And the same construction is now the driver for a different arm**: it is
exactly how `a_vem_whose_per_field_graph_cannot_open_names_what_it_takes_down`
produces a per-field graph failure — a one-byte edit turning a graph-less
field's `vectorIndexLength = 0` into `1`.

## F1 `[CORRECTNESS → fixed]` a `.vemf` region offset sliced the `.vec` with a guard that formed its own sum

**Where**: `crates/lucene-codecs/src/vectors.rs`, `RawVectorValues::ord_to_doc`
and the `doc_to_ord` half of `vector_values_common!`.

```rust
let start = *addresses_offset as usize;
let end = start + *addresses_length as usize;
if end > self.file.len() { return corrupt("...past end of .vec"); }
```

Both operands are read off `.vemf` and **neither was validated anywhere**:
`read_field_entry` proves the `size * dim * byteSize == vectorDataLength`
identity for the *vector data* region and bounds it against the file, but the
two sparse `ordToDoc` regions got neither treatment — nothing on the wire
relates them to the `.vec` they address.

This is `docs/arithmetic-gate.md`'s named shape, *"`if a + b > len` where `a`
came off disk: the guard forms the very sum it exists to guard."* A negative
`addressesOffset` arrives as `usize::MAX` through `as usize`, the addition
wraps to something small, the comparison **passes**, and the slice panics with
`start > end`. The decoder is on the query path
(`KnnFloatVectorQuery`/`KnnByteVectorQuery` and every merge's flat vector
source), so through the FFI this is a dead JVM.

**Fixed**: one `file_region(file, offset, length)` helper — `usize::try_from`
on each operand, `checked_add`, `slice::get` — used by both sites, with the
reported error naming both numbers and the file length. Test:
`no_re_signed_vemf_field_overwrite_crashes_the_vector_checks`, which aborts
without the fix.

## The two items c28 handed to this file's owner

### The `FixedBitSet` crate rule

`check_index.rs` had **four** instances of the forbidden shape, all of the
`deletes::mark_deleted` variety — correct today, one line away from not being:

| site | the index's bound | the bitset |
|---|---|---|
| `check_term_vectors`' per-document loop | `si.doc_count.min(reader.max_doc())` | `.liv` |
| `check_postings`' per-posting loop | `si.doc_count` (after the doc-ID range check) | `.liv` |
| `check_field_norms`' terms-vs-norms cross-check | `si.doc_count` | `.liv` |
| `check_soft_deletes`' live-value count | `keys.len()` | `.liv` |

All four now go through `is_live` / `is_live_at`, which take the bound from
`bits.len()`. An out-of-range doc id reads as **live**, deliberately: this is a
verifier, an out-of-range id is already reported by whatever produced it, and
quietly calling it deleted would suppress every check that only looks at live
documents. `a_doc_id_past_the_end_of_the_live_docs_bitset_is_treated_as_live`
covers both failure modes c28's review named — an *empty* bitset (a real panic,
`words` being empty) and a *short* one (a silent ghost bit) — plus the negative
id whose `as usize` sign-extends.

(The same rule turns up twice more in the *merge* path; see A2 below. That is
the finding underneath this table: the shape recurs wherever a doc id and a
`.liv` come from two different files, and this batch found it in three modules.)

### `MAX_POSITION`

Was duplicated between `indexing_chain.rs` (writer) and `check_index.rs`
(reader). `check_index.rs` now imports the writer's `pub const`. The direction
matters: the two halves of one rule — what `advance_position` clamps to and
what `postings.positions_valid` rejects — must move together, and a verifier
holding its own copy of the writer's ceiling is a verifier that can silently
stop agreeing with the writer it verifies.

### The pre-existing clippy warning

`patch_kdm_doc_count(kdm: &mut Vec<u8>, ..)` → `&mut [u8]` (`clippy::ptr_arg`).
It only indexes and re-signs in place; the `&dyn Fn(&mut Vec<u8>)` the fixture
builder takes is unchanged, because deref coercion at the call site is enough.

## Driven: the arms that can fire

Twelve tests. `check_index.rs`'s **production** uncovered region starts went
**128 → 46**; the arms behind them, by family:

- **`every_listed_file_that_goes_missing_is_reported_by_its_own_open_check`** —
  a table of 21 (fixture, extension, expected check) rows covering
  `.fdt`/`.fdx`/`.fdm`, `.tvd`/`.tvx`/`.tvm`, `.dvm`/`.dvd`/`.dvs`,
  `.nvd`/`.nvm`, `.kdm`/`.kdi`/`.kdd`, `.tip`/`.tmd`/`.doc`/`.pos`/`.pay`,
  `.vemf`/`.vem`. c25 drove two of these; every other one sits *inside* a
  `(|| -> Result<..> { ... })()` block whose failure arm is the only thing
  between a missing file and a silently empty walk. Sixteen arms, one table.
- **`commit_level_degradations_are_reported_rather_than_panicking`** — a
  directory with no `segments_N` at all, a `.si` the commit lists and the
  directory lacks, and a `SegmentInfos` whose generation cannot be turned back
  into a file name (without the fallback, a negative generation is a panic
  *inside the verifier*).
- **`a_byte_encoded_vector_field_is_checked_through_the_byte_branch`** — c25's
  carry-over. Every `.vec` in the repo is `Float32`, so the whole
  `VectorEncoding::Byte` arm had never run. Three cases against one writer,
  differing in exactly one claim each: the healthy control (which also drives
  `check_hnsw_graphs`' silent return for a flat, graph-less segment); a `.fnm`
  whose dimension disagrees with the `.vemf`'s, which the byte branch reports
  through a **length comparison** where the float branch reports a decode
  error; and a `.vec` whose ord→doc map points past the `.si`'s `maxDoc`,
  `check_ord_to_doc`'s only reachable *disagreement* arm.
- **`no_re_signed_vemf_field_overwrite_crashes_the_vector_checks`** — a new
  negative control: each 8-byte-aligned window of the entry region overwritten
  with `i64::MIN`, `-1`, `i64::MAX` and `1 << 40`, footer re-signed, a *report*
  required. It found F1 on its first run.
- **`a_vem_whose_per_field_graph_cannot_open_names_what_it_takes_down`** — the
  **per-field** `hnsw.open:<f>` arm, the one with a `skip_families` call
  attached; a segment can carry several vector fields and lose the graph of
  exactly one. The test asserts the other half too — the flat vector store is
  still fully checked — so the skip is targeted rather than a blanket give-up.
- **`a_re_signed_kdd_corruption_is_reported_by_the_points_decode`** and
  **`a_multi_dimension_points_field_checks_its_leaf_boxes_and_its_doc_count`**
  — the `points.decode:<f>` arm and its four dependent families, plus
  `points.leaf_bounds_subset_of_field`, which is skipped outright for a
  single-index-dimension field and so had never run against anything. It
  matters because a `PointRangeQuery` prunes whole subtrees on the *leaf's own*
  bounding box without reading a point. The same test drives
  `docCount > pointCount` through a new `patch_kdm_doc_count` helper.
- **`a_segment_with_deletions_still_checks_its_vectors_and_norms`** — three
  places consult `live_docs` while walking a doc id from a different file, and
  only the postings one had been reached: no fixture carried both deletions
  **and** term vectors or norms, which is what a real index looks like after
  its first delete. The test proves the walks really saw the deletion (the
  live-only statistics must be strictly smaller than the undeleted control's)
  rather than merely not failing.
- **`no_re_signed_dvm_corruption_of_a_doc_values_index_goes_unnoticed`** and
  **`..._of_a_single_valued_sorted_set_goes_unnoticed`** — c25 named `patch_dvm`
  as the clearly-shaped next block. A **sweep** is the better shape, and the
  reason is worth recording: the arms are not one invariant but a family (a
  per-document decode that fails, an ordinal outside the terms dictionary, a
  non-monotonic SORTED_SET ordinal run, a dictionary whose decoded size
  disagrees with `valueCount`), and every one needs a `.dvm` that *parses* and
  then disagrees with its `.dvd`. A byte flip in the entry region produces
  exactly that; a typed editor would need one entry layout per doc-values type
  to reach the same set.
- **`a_sort_or_soft_deletes_field_whose_values_cannot_be_read_is_reported`**
  and **`the_sort_key_readers_report_every_way_their_input_can_fail`** —
  including that `check_soft_deletes` opens the `.liv` a **second** time,
  independently of `check_live_docs`, because the count is over *live*
  documents; a soft-delete count computed as if nothing were deleted is the
  number `IndexWriter` uses to decide a segment is fully deleted and can be
  dropped.
- `sorted_index` — a Java-written index that declares an index sort — joins
  `every_real_lucene_index_fixture_passes_every_check`, so `check_index_sort`'s
  success path is proved against Lucene's own sort-on-flush output.

## Per-file rejection rates (extending c19's and c25's table)

Every row is a **re-signed** sweep: the footer is recomputed over each
corruption so `file:*`'s CRC cannot claim the catch and only semantic checks
can fire.

| control | corruptions | rejected by the named check | by another check | accepted |
|---|---|---|---|---|
| `.nvm` → `norms.*` (c19) | 99 | **85** | 0 | 14 |
| `.tip` → `postings.seek_agrees` (c19) | 99 | **44** | 12 | 43 |
| `.vex` → `hnsw.neighbors_*` (c19) | 318 | **138** | 3 | 177 |
| `.dvd` sorted → `doc_values.*` (c19) | 99 | **18** | 27 | 54 |
| `.dvd` numeric+binary → `doc_values.*` (c19) | 261 | **69** | 0 | 192 |
| `.doc` → `postings.advance_agrees` (c19) | 2 034 | **5** | 2 026 | 3 |
| `.fdt` → `stored_fields.every_doc_decodes` (c25) | 47 | **33** | **0** | 14 |
| `.tvd` → `term_vectors.every_doc_decodes` (c25) | 43 | **15** | 21 | 7 |
| **`.vemf` (byte, sparse) → `vectors.*`** (c30) | **616** | **566** | **0** | 50 |
| **`.kdd` → `points.decode`** (c30) | **200** | **2** | **198** | **0** |
| **`.dvm` `doc_values_index` → `doc_values.*`** (c30) | **520** | **315** | **0** | 205 |
| **`.dvm` `multi_valued_dv_index`** (c30) | **380** | **225** | **0** | 155 |
| **`.dvm` `sorted_dv_index`** (c30) | **218** | **91** | **0** | 127 |
| **`.dvm` single-valued SORTED_SET** (c30) | **624** | **256** | **0** | 368 |

Three of the new rows are findings in their own right.

- **`.vemf`: 0 caught elsewhere.** Nothing else in the module reads it, so a
  corruption the vector checks miss is one nothing catches — the same
  structural fact c19/c25 recorded for `.nvm`, `.dvd` and `.fdt`. It now has
  four company rows, and the pattern is general: **every metadata file in this
  index format has exactly one reader in the verifier.** The 50 accepted are
  fields whose overwritten value still describes a well-formed region.
- **`.kdd`: 200 of 200 rejected, only 2 by the decode walk itself.** The
  opposite of every other row, and the interesting one: a BKD tree carries
  three independent redundancies over the same bytes (the field's declared
  min/max packed value, its `docCount`, its `pointCount`), so almost every
  flipped byte is caught by a *cross-check* rather than by failing to decode.
  Redundancy on disk is what makes a verifier strong, and points is the only
  subsystem here that has three copies of it.
- **`.dvm`: 0 caught elsewhere, on all four fixtures.** The `.dvm` is pure
  metadata with no second copy, which is why `Lucene90DocValuesProducer`
  verifies the whole file at open — and why a *skipped* `doc_values` family is
  the column's only reader, exactly as c25's `Outcome::Skipped` model assumes.

Every row asserts its `caught_by_other` value, so the claim stays true if a
second reader is ever added.

## Verdict — item 1

**`crates/lucene-index/src/check_index.rs`** — swept clean. 97.25% →
**98.58%** lines. 41 previously-unfired arms driven, 1 more deleted with the
proof c25 asked for, 5 kept as total error handling with the proof at the site
(closing c25's last open D-list question), the `FixedBitSet` crate rule applied
to all four sites, and one CORRECTNESS panic fixed in a decoder that is also on
the query path.

**`crates/lucene-index/src/checksum_verify.rs`** — swept clean at **97.03%**,
and the number is worth one sentence rather than a chase: of its 10 uncovered
lines, **nine are `assert!(cond, "…{:?}", x)` message closures inside its own
test module**, which by construction only evaluate when a test fails, and the
tenth is a closing brace llvm-cov starts a region on. There is no unexercised
production path in that file at all.

**`crates/lucene-codecs/src/vectors.rs`** — not swept (it is c31's); one
CORRECTNESS panic fixed in the two sparse `ordToDoc` region computations, with
a negative control that aborts without the fix. c31 has since audited the rest
of that module.

---

# Item 2 — the arithmetic gate, burned down across the last three modules

`docs/arithmetic-gate.md`'s table goes **3 → none**. With c24/c27/c31 having
finished `lucene-codecs` and c19 `lucene-store`, **the whole workspace is now
audited**: `python3 scripts/check-arith-allows.py` reports
`ok (0 module(s) still unaudited)`.

## Burn-down

| | count |
|---|---|
| modules carrying the marker after c28 | 3 |
| **audited this batch** | **3** |
| lint diagnostics resolved (lib, non-test) | **66** |
| lint diagnostics opted out at a test-module boundary | 2 |
| **remaining marked in the workspace** | **0** |

| Rust file | lib diagnostics | resolved by | Java counterpart (10.5.0) |
|---|---|---|---|
| `merge_policy.rs` | 31 | 12 `checked_*`/`saturating_*`/`try_from`, 6 `// ARITH:` proofs, 13 removed by two new accessors | `index/TieredMergePolicy.java`, `index/MergePolicy.java` |
| `index_writer.rs` | 21 | 1 fix, 20 `// ARITH:` proofs | `index/IndexWriter.java`, `index/IndexingChain.java` |
| `merge.rs` | 14 | 14 `// ARITH:` proofs (plus 2 fixes the lint did not report) | `index/SegmentMerger.java`, `index/MultiSorter.java` |

Per `docs/arithmetic-gate.md`, the audit is three parts and clippy is only the
first. **Both of this item's process-kill findings came out of part 2 — the
hand-check of indexing, slicing and allocation — not out of the lint.** Part 3
(a re-signed byte-flip sweep of the files the module parses) does not apply
here and it is worth saying why rather than skipping it silently: none of these
three modules parses a file format of its own. They consume readers that
`segment_info`, `segment_infos`, `live_docs`, `stored_fields`, `blocktree`,
`doc_values`, `norms`, `points` and `vectors` have already validated, each of
which carries its own sweep. What this batch found instead is the *seam*: two
places where a value one of those readers returned was trusted further than
that reader had validated it.

---

## A1 `[CORRECTNESS → fixed]` the merge sized itself from the `.fdm`, where Java uses the `.si` (**verified SIGABRT**)

**Where**: `crates/lucene-index/src/index_writer.rs`, `execute_merge`;
consumed by `crates/lucene-index/src/merge.rs`, `merge_segments` and
`build_doc_id_maps`.

Java's `SegmentMerger` works from `SegmentReader.maxDoc()`, which is
`SegmentInfo`'s document count — the `.si`. This port took it from the
stored-fields reader instead, and `stored_fields::open` checks only that its
`.fdm` `maxDoc` is non-negative. The module's own comment says so at the site:

> Java takes `numDocs` from `SegmentInfo.maxDoc()`, an already-validated value
> … This port has no `SegmentInfo` to hand, so the `.fdm` copy *is* the
> document count.

That is true of the *codec*, which genuinely has no `SegmentInfo`. It is not
true of `execute_merge`, which opens the `.si` eleven lines later.

`merge_segments` then sizes two `Vec`s per source from that number — the live
doc-id list (`for doc_id in 0..max_doc`, pushing) and `build_doc_id_maps`'
dense `vec![-1i32; max_doc]`. A **four-byte edit** to one `.fdm` claiming
`maxDoc = i32::MAX` therefore reserves ~8.6 GB twice. Under
`( ulimit -v 4000000; cargo test … )`, per `docs/arithmetic-gate.md`'s
reproduction note, the unfixed code gives:

```
memory allocation of 4294967296 bytes failed
... (signal: 6, SIGABRT: process abort signal)
```

An allocation abort is the one failure `catch_unwind` at the FFI boundary
cannot intercept, and a merge is something an `IndexWriter` runs *on its own*
— no caller asked for it.

`build_doc_id_maps` even carried an `// ARITH:` comment claiming `max_doc` was
"a document count from an already-validated stored-fields reader". This is the
failure mode c19's Tier-2 review named — a proof with a correct conclusion and
wrong reasoning — and it is why the comment now names the caller that supplies
the bound instead of asserting one that did not exist.

**Fixed** in `execute_merge`: the `.si` is opened *before* anything is sized by
a document count, `stored_fields::open`'s `maxDoc` is cross-checked against
`si.doc_count`, and a disagreement is `Error::SegmentDocCountMismatch` (mapped
to `FfiStatus::Io` in `lucene-ffi`, the same class as
`UnreadableSegmentPostings`: a segment that contradicts itself, which no caller
argument could make succeed). The `.liv` parse now takes its `max_doc` from the
`.si` too, which is what `open_segment_for_deletes` two thousand lines below
had been doing all along — the asymmetry between those two functions is what
made this findable.

**Test**: `a_segment_whose_fdm_disagrees_with_its_si_about_max_doc_is_not_merged`,
which patches the `.fdm`'s `maxDoc` in place and re-signs the footer so only
the semantic disagreement can fire, with a two-segment merge as the control
first (so the failure is the edit and not the fixture). Verified to SIGABRT
against the unfixed code.

The remaining exposure is a `.si` that itself lies about its document count.
Java does not defend against that either (`SegmentInfo`'s constructor checks
only `maxDoc >= 0`), so this stops where Lucene stops, and `check_index`'s
`stored_fields.doc_count_matches_si` is the check that catches it.

## A2 `[CORRECTNESS → fixed]` the merge indexed a `.liv` bitset with a bound from the `.fdm`

**Where**: `crates/lucene-index/src/merge.rs`, `merge_segments`.

```rust
for doc_id in 0..max_doc {
    let is_live = source.live_docs.map(|bits| bits.get(doc_id as usize)).unwrap_or(true);
```

`docs/arithmetic-gate.md`'s crate rule: **never index a `FixedBitSet` with an
index bounded against anything other than that bitset's own `len()`.** Here the
bound comes off the `.fdm` and the bitset off the `.liv` — two independent
files — and `FixedBitSet::get` does `words[index >> 6]` behind a bare
`debug_assert`. A `.liv` a few bits short reads a **ghost bit** in release: a
document silently merged that had been deleted, or dropped that had not. One 64
or more bits short is an index panic in release as well as debug.

This is the third module in which c28's rule has now found the shape, after
`deletes`/`term_delete` (c28) and `check_index` (item 1 above).

**Fixed**: one length check per source before the loop —
`Error::LiveDocsLengthMismatch` naming both numbers — which is both the bound
the loop needs and cheaper than the per-document `min` that would otherwise
hide the disagreement. Reported rather than clamped, because a merge that
guessed here would write a segment containing documents that were deleted, or
missing documents that were not.

**Test**: `a_live_docs_bitset_shorter_than_max_doc_is_reported_not_indexed_past`,
covering both failure modes the rule names — an **empty** bitset (a real index
panic in release; in debug `FixedBitSet::get`'s own `debug_assert` fires first,
and either way the unfixed test aborts) and one 45 bits into 50 (the silent
ghost bit) — plus the positive control, a bitset that *does* cover `maxDoc`,
whose one deletion is honoured.

A1's cross-check makes this unreachable from `execute_merge`; the check stays
because `merge_segments` is a `pub` entry point whose caller assembles the
reader and the bitset independently, which is precisely the pairing the rule
exists for.

## A3 `[CORRECTNESS → fixed]` `target_search_concurrency as i64` inverted the document budget

**Where**: `crates/lucene-index/src/merge_policy.rs`, `max_allowed_docs` and
`find_merges_excluding`.

`target_search_concurrency` is a `pub usize` on `MergePolicyConfig`.
`usize::MAX as i64` is `-1`, so `max_allowed_docs(100, 0, usize::MAX)` returned
**−100** where Java's `Math.ceilDiv(100, Integer.MAX_VALUE)` is `1`. A negative
document budget is one every candidate merge exceeds, so the policy silently
stops proposing merges for the whole index — and `i64::MIN.div_euclid(-1)` is
a panic in release as well as debug, the one overflowing division there is.

**Fixed**: `i64::try_from(..).unwrap_or(i64::MAX).max(1)` at both sites, which
keeps the divisor positive — the invariant both the `div_euclid` and the
ceiling adjustment already assumed. The `+ 1` that forms the ceiling now
carries a proof that actually holds at its boundary: the quotient equals
`i64::MAX` only when `live == i64::MAX` and the divisor is `1`, and in that
case `rem_euclid` is `0` and nothing is added.

**Test**: `a_target_search_concurrency_above_i64_max_does_not_invert_the_doc_budget`
— fails against the unfixed code with `left: -100, right: 1`.

## A4 `[CORRECTNESS → fixed]` a segment size above `i64::MAX` became a *negative* byte count

**Where**: `crates/lucene-index/src/merge_policy.rs`, `do_find_merges` and
`find_forced_merges`.

`SegmentStat::size_bytes` is a `pub u64`, and the module's signed byte budgets
(Java's are `long`s) reached it through a bare `as i64`. Above `i64::MAX` that
is negative, and every budget then treats the largest segment in the index as
one that costs nothing: it passes `bytes_this_merge + seg_bytes > max_merged`,
is packed into a merge, and `tot_index_bytes` goes *down* where it should go
up. In `find_forced_merges` the same cast decided whether a delete-free giant
is dropped as "already over the max size" — `-1 >= maxMergeBytes` is false, so
it was kept and bin-packed instead.

Two such segments make `bytes_this_merge + seg_bytes` overflow outright
(`i64::MIN + i64::MIN`), which is a **panic in a debug build**.

**Fixed**: two accessors on `SegmentSizeAndDocs`, `size_i64` and
`raw_size_i64`, clamping with `try_from` — so "absurdly large" keeps meaning
large rather than becoming small — and `checked_add` at both packing loops,
where `None` takes the "too large" branch, which is unambiguously the right
answer for a sum that does not fit. `docs/arithmetic-gate.md` names this shape:
*the guard forms the very sum it exists to guard*, and here it guards the two
**hard** bounds (`max_merged_segment_size`, `allowed_doc_count`) whose breach
produces an oversized merged segment rather than a worse heuristic.

The same treatment covers `max_merged_segment_size as i64` and
`floor_segment_size as i64`, both `pub u64` configs: a negative `max_merged` is
a bound `bytes_this_merge` can never be under, so the packing loop would
propose *nothing at all*, silently, for every index.

**Test**: `a_segment_size_above_i64_max_is_treated_as_huge_not_as_negative` —
panics against the unfixed code at `bytes_this_merge + seg_bytes`, and asserts
both halves (the natural merge treats the giant as its own too-large merge; the
forced merge drops it).

## A5 `[CORRECTNESS → fixed]` the norms field length wrapped where Java's throws

**Where**: `crates/lucene-index/src/index_writer.rs`, `build_norms_output`.

```rust
lengths[entry.doc_id as usize] += entry.term_freq() as u32;
```

`lengths[d]` is Java's `FieldInvertState.length` — the per-document field
length a norm encodes. `IndexingChain.PerField.invert` steps it with
`Math.addExact(invertState.length, 1)`: Java **throws** rather than wrapping
past `Integer.MAX_VALUE`. The `+=` here wraps in a release build, and a wrapped
length encodes to a *small* norm — the longest document in the index scoring as
one of the shortest, silently, in every BM25 query over the field. It also
trips `small_float::int_to_byte4`'s own `debug_assert` on the way past
`i32::MAX`.

**Fixed**: `accumulate_field_length`, saturating at `i32::MAX`. Saturation is
the closest honest analogue of Java's throw here, and this is the exception the
gate's "never `saturating_*` as a reflex" rule allows rather than a violation
of it: `int_to_byte4` is a lossy 8-bit quantisation whose top bucket already
encodes everything near `Integer.MAX_VALUE` as `255`, so a saturated length and
Java's exact one produce **the same norm byte for every input Java accepts at
all** — and the values past that are ones Java refuses to index rather than
ones it scores differently. A negative `term_freq` (reachable only from an
occurrence count above `i32::MAX`, since `term_freq()` is
`occurrences.len() as i32`) means "longer than the longest", not "shorter than
none", and is treated as such.

**Test**: `a_field_length_past_integer_max_value_saturates_instead_of_wrapping`,
including that the encoding stays monotonic across the clamp. **This is the one
fix in this batch without a test that fails against the unfixed code**, and the
reason is c28's F7 precedent: the test exercises a helper that did not exist
before the fix, and the reachable-input path needs a single document with ~2
billion tokens (a multi-gigabyte field value) which is not constructible under
the container's 8 GiB cap. Stated here rather than glossed.

## A6 `[defensive → fixed]` a `.kdm` width sliced a `.kdd` packed value

**Where**: `crates/lucene-index/src/merge.rs`, `merge_point_streams`.

`key(p) = &p.1[..bytes_per_dim]` — `bytes_per_dim` comes off the `.kdm` and the
packed value off the `.kdd`, and nothing on the wire relates them beyond
`merge_points`' shape check. Now `get(..)`, with a short value read as "not
sorted", which falls through to plain concatenation — `points::write` sorts in
that case, so this is a cost choice and never a correctness one, and never a
panic in the middle of a merge. Classified defensive rather than CORRECTNESS
because `merge_points` does check `field_meta.bytes_per_dim` and `num_dims`
against the merged field, and `points::check_config` establishes
`num_dims >= 1`, so no reachable input produces the short value today.

## Proofs, and the ones that had to be replaced rather than written

The remaining 40-odd sites carry a tightly-scoped `// ARITH:` proof. Four
families are worth naming because the proof is the interesting part:

- **RAM accounting** (`index_writer`, 8 sites). Every term is a `size_of` or
  the `capacity()` of an allocation this process *currently holds*, so the sum
  is bounded by the address space. `IndexWriter::ram_bytes_used` qualifies only
  because it is reset to `0` on every flush and therefore only ever totals the
  currently buffered documents — a proof that would be false for a
  monotonically-increasing counter, and the reset is what makes it true.
- **Generations** (`index_writer`, 7 sites). c28's `segment_infos::MAX_GENERATION`
  (`i64::MAX / 2`) is enforced on the way in by `parse` *and* on the way out by
  `write`, so every value this writer can hold is at most half the range and
  reaching `i64::MAX` would take 2^62 further commits. Both halves are needed:
  the increment happens before the write, so it is the *parse* cap that bounds
  the value the increment starts from.
- **`committed_doc_count`'s sum** (`index_writer`). Each term is in
  `0..=i32::MAX` (`segment_info::parse` rejects a negative), and the number of
  terms is bounded by `segments_N`'s own parse, which rejects a segment count
  above the bytes left in that file. The product is below 2^62.
- **Run detection** (`merge`, 6 sites, twice over — stored fields and term
  vectors). `doc_order`'s doc ids are drawn from each source's own
  `0..max_doc`, so `doc_id + 1 <= i32::MAX`; `to_doc` steps only while
  `doc_order[j]` *is* `(src_idx, to_doc)`, i.e. while `to_doc` is itself one of
  those ids, so the bound holds on every step rather than only the first.

Where a proof would have had to rest on "nobody has two billion segments", the
accumulator is `saturating_*` instead, with the reason at the site: every one
of them is a merge-policy *budget*, `segments` is a caller-supplied slice this
module does not bound, and a saturated budget can only make the policy merge
less — the hard caps (`max_merge_at_once`, `max_merged_segment_size`) are
enforced by the `checked_add`s of A4, not by these.

## What the `cast_sign_loss` one-shot found

Per c27's recommendation, `clippy::cast_sign_loss` was assessed once per
module during the burn-down rather than adopted. The live defects it points at
here are A3 and A4 — both are `as i64` on an unsigned value rather than the
`as usize` shape the lint is named for, so the lint itself does not report
them; what the audit needed was the *rule* behind it (a sign-changing cast on a
value the module does not own), applied by hand.

---

# Gates and measurements

## Coverage

Measured with `cargo llvm-cov --no-fail-fast -p lucene-index`, run into a
**private `CARGO_TARGET_DIR`** (`target-c30-cov`) after `cargo llvm-cov clean
--workspace` on it — `cargo llvm-cov` merges every `*.profraw` under the target
dir, and c19 and c25 both recorded a run poisoned by a concurrent batch.

| file | c19 | c25 | c30 (`--lib`) | bar |
|---|---|---|---|---|
| `check_index.rs` | 89.19% | 97.25% | **98.58%** (113 missed of 7 930; regions 98.22%) | ✅ |
| `checksum_verify.rs` | 97.11% | 97.11% | **97.03%** (10 missed of 337; 9 of them test-module assert closures) | ✅ |
| `index_writer.rs` | — | — | **98.45%** | ✅ |
| `merge.rs` | — | — | **98.55%** | ✅ |
| `merge_policy.rs` | — | — | **99.30%** | ✅ |

Whole-crate `--lib`: **98.49%** lines over 34 452, 676 tests, 0 failures.

The crashed attempt's report carried a carry-over asking for the same
measurement under `--lib --tests`, on the theory that c23's
`tests/positions_write_path.rs` was already covering some of
`check_one_vector_field`'s uncovered region starts and a `--lib`-only run was
undercounting. **It is not: the two runs are identical**, 113 missed lines and
98.58% either way (the whole-crate line total moves by 4, in three other
files). So the residual arms listed below really are uncovered, and the
carry-over is closed rather than inherited.

## Runtime

**Of the verifier: 0.18 s.** `every_real_lucene_index_fixture_passes_every_check`
— `check_directory` over every Java-written fixture in the repo, now including
`sorted_index` — in release, against c25's 0.23 s and c19's 0.28 s. Nothing was
added to any per-document, per-term, per-ordinal or per-byte loop; D11's
deletion removes work, `is_live` adds one `usize::try_from` and one `len()`
comparison per live-docs consultation (both folded into the bounds check
`FixedBitSet::get` was going to do anyway), and F1's `file_region` is two
conversions and one `checked_add` per *field*.

**Of the write path**, on the two figures this batch was told not to regress:

| benchmark | before | this batch |
|---|---|---|
| `index-bench` `index[freqs]` | ~21 µs/doc | **21.53 µs/doc** (21 529.705 ns, 50 000 docs) |
| `merge-bench` stored fields, BULK | c22's fast path | **532x** (365.7 ms → 0.7 ms, 4 × 20 000 docs) |
| `merge-bench` term vectors, BULK merge | — | **660 577x** (0.4 ms) |
| `merge-bench` postings k-way merge | — | 10.6x |

A1's cross-check costs one extra `stored_fields::open` per source per merge —
a `.fdm` metadata parse, once, against a merge that reads every document —
and A2's costs one `len()` comparison per source. Neither is in a loop.

## Gates

- `cargo fmt --all --check` — clean.
- `cargo clippy --workspace --all-targets -- -D warnings` — clean. (It was red
  on exactly one diagnostic when this batch started, the pre-existing
  `ptr_arg` at `check_index.rs:11824`, now fixed.)
- `python3 scripts/check-arith-allows.py` — **ok (0 module(s) still
  unaudited)**. `docs/arithmetic-gate.md`'s table now reads "on, fully
  audited" for all three gated crates.
- `python3 scripts/check-java-refs.py` — ok, 242 citations verified against the
  pinned 10.5.0 tree.
- `python3 scripts/check-parity.py` — ok.
- `cargo test -p lucene-index` — 676 lib tests + every integration binary,
  0 failures.
- `cargo test -p lucene-ffi` — 523, 0 failures (A1 adds an `Error` variant that
  `lucene-ffi`'s exhaustive `match` had to learn).
- `scripts/verify-write-path.sh` — **22/22**, confirmed by running it.
- **`scripts/docker-test.sh gate` → `gate: ok`**, every step: fmt, clippy
  (x86_64 and aarch64), check-arith-allows, check-parity, check-java-refs, and
  `cargo llvm-cov --workspace --fail-under-lines 95` at **98.10%** workspace
  lines with **no file below 95%**.

One cross-batch note, since c31 asked: the brief red it saw in `merge.rs`
(`LiveDocsLengthMismatch` used before declaration) was this batch's A2 landing
in two steps. It is resolved — the variant is declared on `merge::Error` and
mapped through `index_writer::Error::Merge`, and `lucene-ffi`'s exhaustive
`match` learned A1's new `SegmentDocCountMismatch` in the same change.

The crashed attempt's own report, `docs/sweep/m2/c30-check-index-arms.md`, is
deleted: everything in it that this batch verified is folded in above with the
measurements its `PLACEHOLDER`s were waiting for, and leaving a second
half-written account of the same work in the tree is the drift this sweep
exists to remove.

## Carry-over

- [ ] **A `.tip` that disagrees with its `.tim`.** Eight arms (three `seekCeil`
      disagreements, the re-seek `Err`, the intersect mismatch) are unreachable
      by byte corruption — c19's 99-corruption `.tip` sweep produced none — and
      need a hand-built trie that resolves to a *different but still valid*
      term. The largest remaining block in `check_index.rs` and the one with
      the clearest shape.
- [ ] **A term-vector fixture with a repeated term across documents**, to drive
      `check_one_vector_field`'s memo *hit*. The memo is why that cross-check
      is O(sum of docFreq) rather than quadratic.
- [ ] **A SORTED_NUMERIC index-sort fixture**, for `sort_key_values`' second
      branch.
- [ ] **`check_field_norms`' empty-field (`docsWithFieldOffset == -2`) path**
      needs a `.nvm` entry for a field no document has a norm for; this port's
      `norms::write_fields` cannot emit one, so it needs `.nvm` surgery.
- [ ] **A `.si` that lies about its document count** is where A1 stops, because
      it is where Lucene stops. `check_index`'s
      `stored_fields.doc_count_matches_si` catches it after the fact; nothing
      catches it *before* a merge sizes itself from it. Worth revisiting if the
      FFI ever exposes a merge over a directory this process did not write.
