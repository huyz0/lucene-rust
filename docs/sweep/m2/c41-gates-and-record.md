# c41-gates-and-record — the gates, and making the record true

A tooling-and-record batch, not a port. Three ledger items: **27** (mechanical
gates for the defect shapes this sweep kept re-finding by hand), **26** (a
rustdoc pass in the gate), **28** (`norms::parse_meta`'s signature), plus the
record-keeping job on `LEDGER.md`'s 16 stale open boxes.

**No Java counterpart exists for the tooling.** Per the protocol's rule 1 no
Java path is claimed for `scripts/`, `docs/` or `.github/`. Two *code* findings
below do have one, and it is cited: `FSTEnum.doSeekFloorArrayDirectAddressing`
and `FSTEnum.findNextFloorArcDirectAddressing`
(`/home/tuong/work/lucene-10.5.0/lucene/core/src/java/org/apache/lucene/util/fst/FSTEnum.java:419`,
`:514`).

The gates found **36 live sites of two defect classes**, and the batch's own
Tier-2 review found 31 of them by pointing out that the first version of the
`fixed-bitset-bound` rule could not see the *dominant* shape. That sequence is
the headline, and it is worth stating in the order it happened:

1. The shapes were already written down in `docs/arithmetic-gate.md`, and three
   separate batches had audited by hand against exactly those words.
2. The mechanical form found five more.
3. The mechanical form was then found to be **blind to roughly half the real
   index sites in `lucene-search`** — every `live_docs.is_none_or(|bits|
   bits.get(doc as usize))`, because `bitset_names` did not follow closure
   parameters. Widening it turned 5 findings into 36 sites.

The lesson is the one `docs/mechanical-gates.md` opens with, one level up: a
gate's *coverage* needs the same scepticism as the code it checks, and the way
to get it is to count what the gate sees against what a grep sees. 36 sites vs
36 "index sites" reported looked consistent until someone counted the greps.

---

## 1. Item 27 — the gates

New: [`scripts/check-port-invariants.py`](../../../scripts/check-port-invariants.py),
six rules, in `scripts/gate.sh` and in CI. Documented, blind spots first, in
[`docs/mechanical-gates.md`](../../mechanical-gates.md).

| rule | what it enforces | sites today |
|---|---|---|
| `fixed-bitset-bound` | a `FixedBitSet` index is bounded by *that bitset's* `len()`, or carries an `// FBS:` proof | 35 index sites, 15 proofs (30 more now go through `FixedBitSet::get_doc`) |
| `sentinel-callers` | a `-1`-returning fn declares `// SENTINEL:`; every call site tests it or carries `// SENTINEL-OK:` | 8 declared fns, 6 call sites, 1 waiver |
| `codec-suffix-literal` | no `LuceneNN_N` literal outside `per_field_codec_suffix` | 0 |
| `blocktree-infallible` | no `seek_exact`/`seek_ceil`/`current` in a `blocktree` consumer | 0 (regression guard) |
| `doc-values-per-doc` | no *new* per-document `doc_values::numeric_value`/`binary_value` | 10, on a keyed burn-down |
| `ledger-single-list` | `LEDGER.md` contains no `- [ ]` | 0 |

Two more gates are not in that script: `check-parity.py` now validates a
`docs/parity.md` row's `::item` suffix, and the rustdoc pass (item 26) covers
`[`linked`]` symbols.

### 1.1 Every gate was watched to fail

The requirement, and the reason for it: this sweep found three checks that
could not fail and one that reported "pass" over a segment it had never opened.
Each rule below was verified by introducing the defect, running the check,
and reverting.

| rule | defect introduced | message |
|---|---|---|
| `fixed-bitset-bound` | reverted c28's fix in `deletes::mark_deleted` (bound on `max_doc`, not `bits.len()`) | 2 violations at `deletes.rs:157`, `:158` |
| `fixed-bitset-bound` (closure shape) | put `soft_deletes::is_live` back to `is_none_or(\|bits\| bits.get(doc as usize))` | 1 violation — the shape the first version of the rule could not see at all (finding 13) |
| `sentinel-callers` (half 1) | deleted `bit_table_next_bit_set`'s `// SENTINEL:` line | `fst.rs:414: returns a bare -1 sentinel with no // SENTINEL: declaration` |
| `sentinel-callers` (half 2) | deleted c31's own `if next_index == -1` check | `fst.rs:1831: call of bit_table_next_bit_set ... with no test of the sentinel` |
| `sentinel-callers` (half 2, this batch's own fix) | deleted finding 1's `if floor_index == -1` check | 1 violation at the `seek_floor_direct_addressing` call site |
| `codec-suffix-literal` | added `const HARDCODED_SUFFIX: &str = "Lucene90_0";` to `field_updates.rs` | `field_updates.rs:1229: a LuceneNN_N ... literal outside per_field_codec_suffix` |
| `blocktree-infallible` | added `f.seek_exact(b"x")` to `query_cache.rs` | `query_cache.rs:637: use the try_ form` |
| `doc-values-per-doc` | added a `for doc in 0..max_doc { numeric_value(..) }` loop to `facets.rs` | `fn new_per_doc_loop ... per document 1 time(s), 0 recorded` |
| `ledger-single-list` | appended a `- [ ]` line to `LEDGER.md` | `LEDGER.md:2069: an unticked - [ ]` |
| rustdoc | added `//! See [`a_function_this_module_does_not_have`]` to `live_docs.rs` | `error: unresolved link ... -D rustdoc::broken-intra-doc-links` |
| `parity ::item` | renamed a `parity.md` row's item to one that no longer exists | `parity.md:52: live_docs.rs does not define write_removed_last_batch` |

### 1.2 What each gate cannot catch

Written out per rule in `docs/mechanical-gates.md`, because c19's arithmetic
gate was accepted precisely on that basis and c25/c27/c31 then found nine
defects in its documented blind spot. The load-bearing ones:

- `fixed-bitset-bound` only checks a site whose receiver it *knows* is a
  bitset. That name detection is the whole rule, and it is where this batch's
  own review found the gate wanting — see finding 13. It also proves only that
  the bound came from the bitset, not that the arithmetic on it is right, and a
  function mentioning `bits.len()` for an unrelated reason satisfies it. (All
  three historical instances had no `.len()` anywhere in the function.)
- `sentinel-callers` cannot tell whether a check is the *right* check
  (`> 0` and `>= 0` both satisfy it and mean different things — see finding 1),
  sees only `-1`, and — measured — checks **6 call sites while ~21 are
  method-syntax and therefore invisible**. Its second half is in practice a
  *declaration* gate; the per-call-site audit for a method-syntax sentinel is
  still by hand.
- `blocktree-infallible` omits `next`, because matching `Iterator::next`
  crate-wide is hopeless. `try_next`'s migration is unguarded.
- `doc-values-per-doc` cannot see a per-document call hidden behind a helper.
- rustdoc only checks symbols inside `[...]`. **A symbol in plain backticks is
  invisible to every gate here**, which is the whole of what item 27(c) still
  asks for; see §6. When an intra-doc link to a private item in another module
  will not resolve, widen the item to `pub(crate)` rather than demoting the
  link to backticks — the second makes the gate green by deleting what it
  checks.

---

## 2. Findings

### Finding 1 — `fst::seek_floor_direct_addressing` passes a `-1` sentinel into a decode

`[CORRECTNESS]` — **fixed**, `crates/lucene-codecs/src/fst.rs`.

Java: `FSTEnum.doSeekFloorArrayDirectAddressing` (`FSTEnum.java:419`) reads
`int floorIndex = BitTable.previousBitSet(targetIndex, arc, in);` and follows it
with `assert floorIndex != -1;` before `readArcByDirectAddressing`.

Rust: no check. `bit_table_previous_bit_set`'s `-1` went straight into
`bit_table_count_bits_up_to` and `read_arc_by_direct_addressing`.

Consequence: `read_arc_by_direct_addressing` derives the arc's label as
`first_label + index`, so `-1` yields `first_label - 1` — an arc one label
*below* the range the node declared. For `first_label == 0` that is exactly
`END_LABEL`. It is a **plausible wrong answer**, not a rejected input, and the
address it computes is inside the node body, so nothing downstream catches it.
`-1` is reachable from a single flipped presence bit.

This is **c31's defect, in the sibling function**. c31 fixed
`read_next_real_arc` and its report claimed both sites were handled; the
arithmetic gate's own write-up says the class needs a per-call-site record.
Nothing enforced that until now, and c31's 40 136-flip byte sweep passed over
this code — as the write-up predicts it must, because a plausible wrong label
is a clean decode.

Resolution: reject `-1` as corruption, with the reasoning inline. Regression
test `a_floor_seek_into_a_gap_with_no_arc_below_it_is_rejected`, built at
`first_label == 0` deliberately — every FST fixture in the tree starts at an
ASCII letter, where the same bug is a harmless-looking off-by-one label.
**Verified to fail without the fix** (it returns `Ok`, a clean wrong decode).

### Finding 2 — `soft_deletes::is_live` sign-extends a negative doc id into `FixedBitSet`

`[CORRECTNESS]` — **fixed**, `crates/lucene-search/src/soft_deletes.rs`.

`live_docs.is_none_or(|bits| bits.get(doc as usize))` on a `pub fn` taking a
caller-supplied `doc: i32` and a caller-supplied `&FixedBitSet`. A negative
`doc` becomes `usize::MAX`; a `doc` merely past the bitset reads a ghost bit and
reports a document **live** that the bitset never covered. This is c28's
`term_delete::resolve_term_doc_ids` defect verbatim, on a different `pub` API.

Resolution: `bits.get_doc(doc)` — "a doc outside the live-docs bitset is not
live", the answer `vector_query`'s accept filter already gave. `get_doc` is new
(finding 13); this site was the first of **thirty** that now share it. Tests
`a_negative_doc_id_is_not_live_rather_than_a_panic` and
`a_doc_past_the_live_docs_bitset_is_not_live`; both fail without the fix.

### Finding 3 — `soft_deletes::clear_present` bounds on `max_doc`, indexes the caller's bitset

`[CORRECTNESS]` — **fixed**, `crates/lucene-search/src/soft_deletes.rs`.

`hard_live_bits` returns `live_docs.cloned()` when there is one — a bitset whose
length is the **caller's**, not `max_doc`. `clear_present` and the overlay loop
then bounded every doc against `max_doc`, a separate parameter. This is c28's
`deletes::mark_deleted` shape, and here the two genuinely can disagree: both are
parameters of a `pub fn`.

Consequence, with a dense soft-deletes entry (`PresentDocs::Every(count)`):
ghost writes into the last word, then an index panic. A soft-delete that clears
a ghost bit is a **live document reported dead** once `cardinality()` — a
whole-word popcount, as Java's is — counts it.

Resolution: `let num_bits = bits.len().min(max_doc);` hoisted, used by both
loops, with the rule spelled out. `hard_live_bits`' doc comment now says its
result's length is the caller's. Test
`a_max_doc_larger_than_the_live_docs_bitset_does_not_write_a_ghost_bit`; it
panics without the fix.

### Finding 4 — `query_cache::search_term_query_cached`, two unrelated bounds

`[CORRECTNESS]` — **fixed**, `crates/lucene-search/src/query_cache.rs`.

Two instances in one function:

- `FixedBitSet::new(num_docs)` indexed by doc ids the postings walk produced.
  `num_docs` is the caller's claim about the segment; the doc ids come from the
  segment's `.doc` file. Now bounded on `bits.len()` with a new typed
  `Error::CachedDocOutOfRange`, because silently dropping a hit is a wrong
  result set and absorbing it is worse than reporting it.
- `live_docs.is_none_or(|bits| bits.get(doc_id as usize))` — a *different*
  caller-supplied bitset from the cached set being iterated. Same fix as
  finding 2 (`get_doc`).

The two halves take **different policies deliberately**, which is worth stating
because they look inconsistent: dropping a hit from the *cached set* changes
the result set and has no defensible answer, so it is an error; a doc the
*live-docs* bitset does not cover has one — "not live" — so it is answered.
The rule is "report when there is no right answer, answer when there is", and
it is the same rule `soft_deletes` and `vector_query` follow.

Tests `a_num_docs_below_the_segments_own_doc_ids_is_reported_not_absorbed` and
`a_live_docs_bitset_shorter_than_the_cached_set_hides_the_docs_it_does_not_cover`;
both panic without the fix.

### Finding 5 — `exact_search` iterates the scorer's ordinals against a foreign bitset

`[CORRECTNESS]` — **fixed**, `crates/lucene-search/src/vector_query.rs`.

`for ord in 0..scorer.max_ord() { if !accept_ords.get(ord as usize) }`.
`max_ord()` is the flat vector store's count; `accept_ords` is whatever the leaf
plan built. Now bounded on `accept_ords.len()`, hoisted out of the loop: an
ordinal the accept set does not cover is, by definition, not accepted. This one
provably drops nothing — `cost` is `accept_ords.cardinality()` over the same
bitset, so the ordinals the bound excludes were never counted. The same
function's live-docs and filter checks now go through `get_doc`.

### Finding 6 — `FixedBitSet::get`/`set`/`clear` only `debug_assert`ed their bound

`[CORRECTNESS]` — **fixed**, `crates/lucene-util/src/fixed_bit_set.rs`.

Java's `FixedBitSet` carries the bound as an `assert`, off in production, and
this port ported the `assert`. `words[index >> 6]` alone catches only an index
64 or more past the end; one merely past `num_bits` lands inside the final word
and reads or writes a **ghost bit**. That is the half of the class that costs a
wrong answer rather than a crash, and it is not something a caller can detect.

Resolution: the bound is checked unconditionally, through a `#[cold]
#[inline(never)]` panic helper so the hot paths (`hnsw`'s `visited` set, one
index per graph node) pay one never-taken branch and no inlined formatting. A
panic is containable — `lucene_ffi`'s `guard` catches it and reports
`FfiStatus::Panic`; a ghost bit is not. Four `#[should_panic]` tests, each using
an index in the ghost range where the slice index alone catches nothing.

This does **not** make the `fixed-bitset-bound` rule redundant: a
wrong-but-in-range index is still a wrong answer.

### Finding 7 — three doc comments describing symbols that no longer exist

`[MISSING]` (record) — **fixed**, found while clearing item 26's links.

- `check_index.rs` referenced `check_postings_term_stats` twice; the function is
  `check_postings` and the name survives only as the *check name*
  `postings.term_stats`.
- `postings_writer.rs` referenced `write_multi_children_root` three times, and
  its module doc described, in the present tense, a multi-block `SIGN_MULTI_CHILDREN`
  `.tip` writer that commit `6f4d20d` **deleted** — because real Lucene cannot
  read it (`SegmentTermsEnum` hands `loadBlock` an `fp` of `-1` for a root node
  with children but no output). A reader planning work off that doc would have
  started by writing code that was removed on purpose. Rewritten to describe the
  single-block shape the writer actually emits, with the reason.
- `index_writer.rs`'s `NormsFieldConfig` doc described the pre-c35 opt-*in*
  world ("single-field-only, a single `Option`, not a list") while
  `norms_field_configs` returns a `Vec` and the only knob is the opt-out
  `omit_norms_field`. It also linked a non-existent `IndexWriter::set_norms_field`.

Two more of the same class were caught by the new `check-parity.py` rule on its
first run: `parity.md` named `index_writer.rs::write_index_sort_to_si` (the code
is in `build_and_write_segment`) and `error.rs::ffi_get_last_error_message`
(it lives in `lib.rs`). Both corrected.

### Finding 8 — a rustdoc trap that silently breaks a module's own links

`[MISSING]` (tooling) — **fixed**, `crates/lucene-codecs/src/lib.rs`.

An outer `///` doc on a `mod` declaration makes rustdoc resolve *the whole
merged doc* — including the module file's own `//!` lines — in the **crate-root**
scope. Every link the module writes to its own items breaks, and the diagnostic
carries **no file and no line**, so it is invisible in a 234-error log.
`direct_reader`, `for_util` and `lz4` all had it. The rationale those `///`
blocks carried now lives in each module's own `//!` header, and `lib.rs` says
why in a comment.

### Finding 9 — CI's fixture job runs on a JDK the fixtures are not built with

`[CORRECTNESS]` (tooling) — **fixed**, `.github/workflows/ci.yml`.

Item 26b said `gen-fixtures.sh --check` "is run by hand". It is not — it has
been a CI job all along. What is wrong is subtler: that job pinned
`java-version: '25'` while `docker/Dockerfile`, AGENTS.md and the committed
fixtures use **JDK 21**, and `--check` byte-compares
`break_iterator/manifest.properties`, which *records the JDK version*. c36 fixed
that fixture by regenerating it under 21; the runner that verifies it stayed on
25. Both Java jobs now pin 21. Verified in the container: 48 deterministic files
byte-identical, 0 mismatches, 0 missing, 0 extras, 0 manifest key-set problems,
0 segment-id disagreements.

The transferable half: **a check that fails for a reason nobody reads is the
same defect as a check that cannot fail** — it just takes longer to be ignored.

### Finding 10 — three gate steps were in `gate.sh` but not in CI

`[MISSING]` (tooling) — **fixed**, `.github/workflows/ci.yml`.

`cargo check --manifest-path benchmarks/rust-runner/Cargo.toml --all-targets`
went into `scripts/gate.sh` in c36 and never into CI — so it ran only for
developers who had installed the hook, which is the exact gap that let 11 clippy
warnings into the tree after a toolchain bump. Added, along with the two new
steps (`check-port-invariants`, the rustdoc pass).

### Finding 11 — `doc_values` per-document re-derivation, ten live sites

`[PERF]` — **recorded with a burn-down**, not fixed.

Ten production call sites call the free `doc_values::numeric_value`/
`binary_value` once per document, re-deriving the column's addressing each time,
where `NumericReader`/`BinaryReader` derive it once. Listed by (file, fn) in
`docs/mechanical-gates.md`. Migrating them is a contained follow-up; the gate
now stops the count growing and fails if it shrinks without the list being
updated, which is what makes it a burn-down rather than a note.

### Finding 12 — `clippy::disallowed_methods` is the wrong shape for item 27(a)

`[INTENTIONAL]`. The ledger proposed a `disallowed_methods` entry on the two
free functions. Measured: 66 call sites, most in test code, and every
single-document lookup is a legitimate use of the API. A deny costs 60-plus
`#[allow]`s whose proofs say nothing — the failure mode
`docs/arithmetic-gate.md` names for a lint adopted too widely. The
loop-detecting burn-down targets the actual defect (per-document
re-derivation) instead. There is still no `clippy.toml` in the tree.

### Finding 13 — the gate could not see the shape it was built for

`[CORRECTNESS]` (tooling) — **fixed**, `scripts/check-port-invariants.py`;
**31 sites fixed** across `lucene-search`, `lucene-codecs` and `lucene-index`.

Raised by this batch's own Tier-2 review, which is the only reason it is here.

`fixed-bitset-bound`'s first version collected bitset names from type
annotations, `FixedBitSet::` constructors, `let` statements and
`cardinality()`/`words()`/`clear_all()` receivers. It did **not** follow a
closure parameter — so `live_docs.is_none_or(|bits| bits.get(doc as usize))`,
the single most common way this codebase indexes a bitset, was invisible. Nor
did it follow `if let Some(bits) = live_docs`. The rule reported 36 index
sites; a grep finds ~64.

Every one of those ~28 unseen sites is finding 2's defect verbatim, on the
*primary* search entry points (`lucene-search/src/lib.rs` ×15,
`doc_value_query.rs` ×8, `field_norms.rs`, `points_query.rs`,
`points_delete.rs`, `hnsw.rs` ×2). Two of them — `points_query.rs:303` and
`points_delete.rs:98` — are c30's shape verbatim: the doc ids come from the BKD
`.kdd` walk, the bound comes from a `.liv`-derived bitset, two different files.

**Finding 6 made this urgent rather than merely wrong.** Making
`FixedBitSet::get` panic in release converted those 28 sites from "silently
wrong live/dead answer" to "aborted query" — a defensible trade, but not one
the batch had audited or tested.

Resolution, in two parts:

1. `bitset_names` now follows closure parameters, `if let Some(x) = ..`,
   `match .. { Some(x) => .. }` and `for x in ..` rebindings of a known bitset,
   to a fixpoint. It reports 37 sites. The `.len()` test also now runs against
   comment-stripped source, so prose cannot waive it (A2, latent, no site
   exploited it).
2. Rather than hand-write the bound thirty times, **`FixedBitSet::get_doc(doc:
   i32) -> bool`** — "is the bit for this externally-supplied id set", `false`
   for a negative id and for one past the bitset. Thirty sites migrated onto
   it, so the rule now lives in one place and the gate has nothing left to
   report. `get_doc` is deliberately not checked by the rule: it *is* the
   bound.

The one remaining site (`hnsw_vectors::new_ord_mapping`) is a `set`, provably
bounded, and carries an `// FBS:` proof naming the caller that makes it so.

### Finding 14 — the `FixedBitSet` hardening invalidated eight comments

`[MISSING]` (record) — **fixed**.

Eight comments across `hnsw.rs` (×4), `check_index.rs`, `merge.rs`,
`docid_set.rs` and `deletes.rs` said some variant of "`FixedBitSet::get` indexes
`words[index >> 6]` behind a `debug_assert`, so an out-of-range index is a ghost
bit in a release build". After finding 6 that is false. Several are the
*justification text* for a bound, so a future reader would have re-derived the
wrong risk model from them — the same defect class as finding 7, created by
this batch. `docs/arithmetic-gate.md`'s own rule statement had it too, and now
records both the new behaviour and why the rule survives it.

### Finding 15 — `RoaringBuilder::add` did not enforce the precondition its proof names

`[CORRECTNESS]` — **fixed**, `crates/lucene-search/src/docid_set.rs`.

`append_in_current_block`'s `// FBS:` proof reads "a doc id this builder has
already accepted, i.e. `doc_id < self.max_doc`". `add` asserted only
monotonicity, and both `add` and `new` are `pub`: `RoaringBuilder::new(100)`
plus a doc in block 1 sizes the dense bitset at **zero bits** and then indexes
it. The proof was a statement about callers that the type did not enforce —
exactly what "review the proof, not its presence" is for, and the reviewer
applied it to a proof written in this batch.

`add` now asserts `0 <= doc_id < max_doc`, with the reason, and two
`#[should_panic]` tests pin it.

### Finding 16 — a dead parameter and five stale test/doc references the link fix missed

`[MISSING]` (record) — **fixed**.

Finding 7 fixed `write_multi_children_root`'s three *linked* references and
missed everything in plain backticks, which is precisely the blind spot §1.2
names — so it is worth recording that the blind spot bit this batch inside the
same file:

- `write_tim_block`'s `strip_prefix_len` parameter was **dead** (its one caller
  passes `0`) and its doc described the deleted multi-block case in the present
  tense. Parameter removed, doc rewritten.
- Two unit tests and one integration test were named and documented for the
  deleted writer (`many_leading_byte_groups_force_multi_child_trie_root`,
  `empty_term_falls_back_to_single_block_...`,
  `term_query_finds_correct_docs_across_multiple_tim_blocks`) and referenced
  `group_terms_by_leading_byte`, which no longer exists. Renamed and re-documented
  around the property that outlived the writer — a field spanning many leading
  bytes still reads back term-for-term — rather than deleted, because that
  property is worth pinning and the tests still pin it.

### Finding 17 — a rewritten doc comment landed on the wrong struct

`[MISSING]` (record) — **fixed**, `crates/lucene-index/src/index_writer.rs`.

The `NormsFieldConfig` doc block finding 7 rewrote sat above
`#[derive(Debug, Clone)]`, which is itself above `SourceDocValueColumns`' own
doc — so rustdoc rendered both on `SourceDocValueColumns`, and
`NormsFieldConfig` had no doc at all. The misplacement predates this batch; the
batch rewrote the text in place, in a pass whose whole purpose was making docs
true of the code, without noticing which item it documented. Moved.

---

## 3. Item 26 — the rustdoc pass

**Closed.** c22 recorded it as blocked on pre-existing broken links; c34 should
have re-measured and did not. The clean-up was **65 links across seven files**
— an afternoon, carried for four batches.

Breakdown of the 65: 50 unresolved links, 13 `[`write`]` ambiguities
(function-vs-macro, fixed as `[`write()`]`), 1 bare URL, 1 redundant explicit
link target. Twelve of the 50 were the invisible span-less kind from finding 8.
Four named symbols that no longer exist — three from finding 7, plus
`facets.rs`'s `dim_count`, which the first pass demoted to plain backticks
(green, but by deleting what the gate checks) and which is really
`SortedSetFacetCounts::adjust_path_count`.

Four more links pointed at items that *do* exist but are private in another
module, where an intra-doc link cannot resolve even under
`--document-private-items`. Those are now `pub(crate)`
(`postings::LEVEL1_FACTOR`, `collector::rank_order`,
`multi_segment::global_term_stats`, `merge::merge_points`) and the links
restored: the cost is a `private_intra_doc_links` warning the gate already
allows, and the gain is that a later rename breaks the build instead of leaving
a stale name in prose no check can see. **Demoting a link to backticks to make
the gate green is the anti-pattern** — it is the one form this doc says is
invisible to everything.

The gate is
`RUSTDOCFLAGS="-D warnings -A rustdoc::private_intra_doc_links" cargo doc
--workspace --no-deps --document-private-items`, in `scripts/gate.sh`, CI and
AGENTS.md.

Two deliberate choices, both recorded in `docs/mechanical-gates.md#rustdoc`:

- `--document-private-items`, because this port's wire-format knowledge lives in
  the doc comments of private decoders and a link that only breaks there is
  still a broken link.
- `private_intra_doc_links` **allowed**, with the count: 166 sites, essentially
  all a public module doc pointing at the private helper that implements what it
  describes. That is the documentation working. Denying it would mean deleting
  166 useful links to satisfy a lint about rendered HTML.

---

## 4. Item 28 — `norms::parse_meta`'s signature

**Closed as INTENTIONAL.** Recorded and deferred three times (b6 #4, c7 F-23,
c15 §F14). The decision is written into `norms::validate_fields`' doc comment so
it is inherited rather than re-litigated.

Both of the entry's premises were wrong:

- **The count is 4, not 23.** Four production call sites under `crates/*/src/`
  (`directory_reader::open`, `check_index::check_field_norms`,
  `index_writer::execute_merge`, `ffi_open_segment`). The other 30 are
  `#[cfg(test)]` round-trips and integration tests, which would never have
  needed a `&FieldInfos` threaded to them. So the cost was never what made it
  wait.
- **The "right moment" c15 named has arrived and does not settle it.** c40's
  validating `FieldInfos` constructor makes the parameter available at three of
  the four sites. It still should not be added.

What settles it:

1. **Norms need no `FieldInfos` to parse.** Every `.nvm` entry is fixed-shape.
   `doc_values::parse_meta` *does* take a `&FieldInfos` — but structurally: a
   doc-values entry carries a skip-index sub-record only when the field's own
   `FieldInfo` says so, so the byte stream cannot be walked without it
   (`doc_values.rs:406-413`). The two signatures differ because the two formats
   do, which is the port's rule 2 (port the wire format, not the class graph).
2. **One of the four sites cannot take the folded form.**
   `check_index::check_field_norms` needs the parse to succeed and *then*
   reports a separately-named `norms.entries_name_real_norms_fields` check, so
   a `.fnm`/`.nvm` disagreement is a named failure with the rest of the norms
   pass (`norms.entry_present`, the per-field values) still running. Folding the
   validation in would abort that pass at the first problem and would still need
   a second, non-validating entry point. It moves the split, not removes it.
3. The behaviour gap has been closed since c15: `validate_fields` runs at both
   places Java's diagnostic fires.

---

## 5. The record-keeping job

`LEDGER.md` carried **16 unticked `- [ ]` boxes, none in the reconciled "Open
work, prioritised" section**. Each was verified against the tree — reading the
code, grepping the symbol, running the check — never against a batch report.

| verdict | count | boxes |
|---|---|---|
| **closed, never ticked** | 7 | items 2 (c36), 13/20/22 (c39), 26/27/28 (c41) |
| **open, duplicate of a live prioritised item** | 9 | items 5 (×2), 12, 14, 15, 18, 21, 24, 25 |

The seven closures were each re-verified in the tree, not taken from the report
that claimed them:

- item 2 — `segment_infos.rs:309/318/330`, all three `advance_*_gen` methods end
  in `generation_advanced()`.
- item 13 — `points.rs:647`, `PointsReader::estimate_point_count`.
- item 20 — `check-port-invariants.py --only=blocktree-infallible` finds zero
  production call sites, which is a stronger check than reading the report.
- item 22 — `indexed_disi.rs:230/317-326`, `jump_table` split off and used.

### Making it structurally undriftable

Prose did not hold this: c34 was run to remove exactly this drift and 16
duplicates survived it, misleading six later batches. So the file now has:

1. **A preamble with one rule** — plan only from "Open work, prioritised";
   everything below is a frozen archive that does not get a second vote — plus
   the two habits the archive's own history argues for (verify against the tree,
   not a report; a recorded blocker is a claim with an expiry date).
2. **A mechanical invariant**: the file contains **no `- [ ]`**, enforced by
   `check-port-invariants.py --only=ledger-single-list`, in the gate and CI.
   Every archive entry is now `- [x]` (closed, naming the batch **and the
   evidence in the tree**), `- [~]` (obsolete) or `- [->]` (open, and naming the
   numbered open-work item that tracks it). Ticking a prioritised item without
   doing one of those three to its archive twin now fails the commit.

The prioritised list's own running tally is updated: c41 closed items 26, 26b
and 28 and all of 27 except (c) and (e), leaving **14 distinct open findings**,
and raised **8d** and **8e** (findings 1 and 2–4 above), both fixed in this
batch.

---

## 6. Verdict

- **Item 27**: closed except sub-items (c) and (e), both restated with the
  measurement that changes their plan. (e)'s proposed grep is provably
  unworkable — of 2 711 committed manifest keys, **2 468 never appear as a
  literal under `crates/`** because the tests build them with `format!()`; the
  shape that works is runtime key-recording, not a grep. (c)'s remaining half
  needs a *diff*-driven check and belongs in `.githooks/pre-commit`, not in
  `gate.sh`, which runs in CI where there is no meaningful diff.
- **Item 26**: closed.
- **Item 28**: closed as INTENTIONAL, with the current call-site count and the
  reason, per the instruction not to defer it a fourth time.
- **Record**: all 16 boxes resolved; zero `- [ ]` remain anywhere in the file;
  a gate now enforces that.

### What the Tier-2 review changed

Recorded separately because the sequence is the batch's most transferable
result. The review (`/quality-review`) was run on the finished diff and
returned **7 gating findings**; five became findings 13-17 above, and the other
two were the `docs/parity.md` rows for this batch's four deliberate divergences
(`FixedBitSet`'s release bound and `get_doc`, `FSTEnum`'s `Corrupt` where Java
asserts, `soft_deletes::is_live`'s `false`-for-out-of-range,
`Error::CachedDocOutOfRange`) — invariant #7, missed entirely.

The single most valuable one was finding 13, and it is the one a self-review
structurally could not produce: **the batch's author had no reason to doubt his
own gate's coverage**, and the gate's own output (36 sites) was internally
consistent. It took someone counting the gate's output against a grep.

### Gates green

```
scripts/docker-test.sh gate                       ok (11 steps)
  cargo llvm-cov --workspace                      98.14% lines, no file below 95%
scripts/docker-test.sh scripts/verify-write-path.sh   ok (23/23)
scripts/docker-test.sh scripts/gen-fixtures.sh --check   ok (48 deterministic byte-identical, 0 mismatches)
check-arith-allows / check-parity / check-java-refs / check-port-invariants   all exit 0
```

### The transferable finding

Two of the three items this batch closed had been deferred on a recorded reason
that had **stopped being true, or had never been measured**: item 26's "needs
the pre-existing broken links cleaned up first" was 65 links and one afternoon,
carried for four batches; item 28's "23 call sites across four crates" was 4
once test code was excluded — and neither number was the reason to decline it
anyway.

That is now the fourth consecutive batch to find this. c39: item 12's recorded
blocker named the wrong thing. c40: item 7b's recorded plan was wrong, and two
items were "blocked on a fixture" that already existed. c41: two recorded
reasons that were never measured. **The recorded reason for deferring an item
is, in this ledger, more often wrong than the finding itself** — which is why
the ledger's new preamble makes "a recorded blocker is a claim with an expiry
date" the second of its two standing habits.
