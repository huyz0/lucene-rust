# Mechanical gates: what they catch, and what they do not

Companion to [`docs/arithmetic-gate.md`](arithmetic-gate.md). That document
covers the one defect class this port has a *lint* for
(`clippy::arithmetic_side_effects`). This one covers the rules it names but
cannot express, plus the two record-keeping rules a Tier-2 review found by eye.
They are enforced by [`scripts/check-port-invariants.py`](../scripts/check-port-invariants.py),
[`scripts/check-parity.py`](../scripts/check-parity.py) and a rustdoc pass, all
three in [`scripts/gate.sh`](../scripts/gate.sh) and in CI. One of them guards a
*document* rather than code -- see [`ledger-single-list`](#ledger-single-list);
it is here because the drift it prevents has cost this sweep more batches than
any single code defect has.

## Why the blind spots are written down first

c19's arithmetic gate was accepted **because** its blind spot -- indexing,
slicing, allocation -- was documented. c25, c27 and c31 then found nine defects
in exactly that blind spot, *because* somebody had written down where to look.
A gate whose limits are undocumented is read as coverage, and coverage is what
stops anyone looking.

So each rule below states what it cannot catch, in the same words it would take
to describe a defect that got past it.

| rule | script | catches | blind to |
|---|---|---|---|
| [`fixed-bitset-bound`](#fixed-bitset-bound) | `check-port-invariants.py` | a `FixedBitSet` index whose enclosing fn never takes that bitset's `len()` | an index bounded by the right `len()` but *derived* wrongly; a bitset reached through a name this file neither types nor rebinds from one it does |
| [`sentinel-callers`](#sentinel-callers) | `check-port-invariants.py` | a `-1`-returning fn with no declaration; a declared sentinel untested at a **free-function** call site | whether the test is *correct*; a sentinel that is not `-1`; **every method-syntax call site** — 6 checked, ~21 not |
| [`codec-suffix-literal`](#codec-suffix-literal) | `check-port-invariants.py` | a `LuceneNN_N` suffix spelled outside `per_field_codec_suffix` | a suffix assembled from pieces (`format!("{fmt}_{n}")`) |
| [`blocktree-infallible`](#blocktree-infallible) | `check-port-invariants.py` | `seek_exact`/`seek_ceil`/`current` in a module that consumes `blocktree` | the same call on a receiver in a module that never names `blocktree` |
| [`doc-values-per-doc`](#doc-values-per-doc) | `check-port-invariants.py` | a *new* per-document `doc_values::numeric_value`/`binary_value` call | a per-document call hidden behind a helper fn; the ten already on the burn-down list |
| [`parity ::item`](#parity-item) | `check-parity.py` | `docs/parity.md` naming a Rust item its own file does not define | prose outside a row's Rust column; an item that exists but no longer does what the row says |
| [`ledger-single-list`](#ledger-single-list) | `check-port-invariants.py` | an unticked `- [ ]` anywhere in `docs/sweep/m2/LEDGER.md` | whether a `- [x]` is *true*, or whether a `- [->]` names the right item |
| [rustdoc links](#rustdoc) | `cargo doc` | a `[`link`]` that resolves to nothing | a symbol named in *plain backticks*, which is most of them |

Between them these six rules cover **the indexing row** of the
arithmetic gate's table (`FixedBitSet` only) and **the two hand-checked rules**
at the end of that document. They do not cover slicing or allocation sizing:
that is still a hand audit, still step 2 of the three-part module audit, and
still where c27 found four aborts and a release-mode infinite loop.

---

## fixed-bitset-bound

**Rule.** Never index a `FixedBitSet` with an index bounded against anything
other than that bitset's own `len()`.

**Why it is mechanical rather than "bound your indices".** The defect always
looks correct locally. It has been found by hand three times:

- c28, `term_delete::resolve_term_doc_ids`: `live_docs.get(doc_id as usize)`
  with no bound at all, and `as usize` sign-extends, so a negative doc id from
  a corrupt `.doc` became `usize::MAX`.
- c28, `deletes::mark_deleted`: the doc id *was* bounded -- against `max_doc`,
  a **separate caller-supplied parameter** from the `&FixedBitSet` it then
  indexed.
- c30, `merge_segments`: the `.liv` indexed by a bound taken off the `.fdm`.

**What the check does.** For every `<recv>.get(..)`/`.set(..)`/`.clear(..)`
where `<recv>` is bound to a `FixedBitSet` anywhere in the file, the enclosing
`fn` must mention `<recv>.len()` or `<recv>.is_empty()`. Otherwise the call
needs an `// FBS:` comment within the 14 lines above it, naming the invariant
that makes the bound sound -- the same contract `// ARITH:` proofs carry, and
the same review obligation: **review the proof, not its presence.**

Test code (`#[cfg(test)]`) is out of scope, for the reason
`docs/arithmetic-gate.md` gives for its own carve-out.

**Name detection, and why it is the whole rule.** A site is only checked if
the checker knows the receiver is a bitset. Names come from type annotations,
`FixedBitSet::` constructors, `let` statements mentioning either, receivers
calling `cardinality()`/`words()`/`clear_all()`, **and rebindings of a known
bitset** -- closure parameters (`live_docs.is_none_or(|bits| ..)`), `if let
Some(bits) = ..`, `match .. { Some(bits) => .. }`, `for bits in ..`.

That last group is not an afterthought. The first version of this rule omitted
it, reported 36 sites, and c41's own Tier-2 review found **31 more it could not
see** -- roughly half the real index sites in `lucene-search`, all of them the
single idiom `live_docs.is_none_or(|bits| bits.get(doc as usize))`. A gate that
misses the dominant shape is worse than no gate. If you extend this rule,
extend `bitset_names`, and check the count moved.

**Blind spots.**

- *A bound taken from the right `len()` but computed wrongly.* The rule proves
  the bound came from the bitset, not that the arithmetic on it is right.
- *A bitset the file never types and never rebinds from one it does.* A bitset
  arriving as an untyped tuple element from another crate is invisible.
- *`len()` mentioned for an unrelated reason.* A function that happens to call
  `bits.len()` somewhere else satisfies the rule without the indexing site
  being bounded. This is the weakest part; it is why the historical instances
  matter -- all three of them had no `.len()` anywhere in the function. The
  `.len()` test runs against comment-stripped source, so prose cannot waive it.
- *`get_doc` is not checked at all*, deliberately: it carries the bound itself.
  The rule's counterpart to a fix is therefore usually "call `get_doc`", not
  "write a proof" -- 30 of the 31 sites the review surfaced were resolved that
  way, in one place instead of thirty.

**A second line of defence, added with the rule.** `FixedBitSet::get`/`set`/
`clear` now check the bound in **release** as well as debug (they were
`debug_assert!` before, which is Java's `assert` and is off in production).
That converts the *ghost bit* -- an index past `num_bits` but inside the final
word, a silently wrong live/dead answer -- into a panic, which `lucene_ffi`'s
`guard` catches and reports. It costs one never-taken, out-of-line branch. It
does **not** make this rule redundant: a wrong-but-in-range index is still a
wrong answer, and no bound check can see it.

## sentinel-callers

**Rule.** A function returning a sentinel *outside* the domain of its result
declares it with `// SENTINEL:`, and every call site tests it.

**Why per call site.** c31 shipped a fix claiming to close one of these. Its
review then found the sentinel still reaching a decode path one function over:
`bit_table_next_bit_set` returns `-1` for "no next present arc", the batch
bounded the *upper* end, and `-1` flowed on as an `arcIdx` so `read_arc`
derived `firstLabel - 1` -- an arc one label below the range the node declared,
and for `firstLabel == 0` exactly `END_LABEL`. The sibling call site had the
check; **the batch's own report claimed both did.** An audit that records "this
function's sentinel is handled" instead of "this call site handles it" will
miss one.

**A byte-flip sweep structurally cannot find this class.** The sweep asserts
"a typed error or a clean decode", and a plausible wrong label *is* a clean
decode. c31's sweep ran 40 136 flips over this exact code and passed.

**What the check does.** Two halves, and the second is what makes the first
non-vacuous:

1. Any `fn` under `crates/*/src/` returning `i8..i64` (bare, or in `Result`/
   `Option`) whose body hands back a literal `-1` **must** carry a
   `// SENTINEL:` line in its doc/comment block. Registration is mandatory, so
   a new sentinel cannot arrive unannounced. There are eight today.
2. Every call of a declared sentinel function must test it within the 22 lines
   that follow -- `== -1`, `!= -1`, `< 0`, `>= 0`, `u32::try_from`,
   `usize::try_from`, `NO_MORE_DOCS`, ... -- or carry a `// SENTINEL-OK:`
   justification.

**Blind spots.**

- *Whether the test is correct.* `if x > 0` and `if x >= 0` both satisfy the
  rule and mean different things. (One of them is right in `fst.rs`'s
  `find_next_floor_arc_direct_addressing` and is Java's own choice; it carries
  a `// SENTINEL-OK:` saying so.)
- *Sentinels that are not `-1`.* `0` for "no such block", `i64::MIN`,
  `u32::MAX` -- none are detected. Declare them by hand.
- **Method-syntax call sites are not checked at all** -- not merely
  cross-crate ones. Calls are matched as bare `name(` or
  `<declaring module>::name(`, so anything reached through a receiver
  (`cursor.doc_id()`, `counts.specific_value(..)`) is invisible, *including in
  the declaring file*. Measured at c41: **6 call sites checked, ~21 unchecked**
  — every checked one is a free function in `fst.rs`/`blocktree.rs`, and
  `find_best_entry_point` has a `// SENTINEL:` declaration with zero enforced
  call sites. Matching bare method names crate-wide produces more noise than
  signal (`doc_id` alone collides with `doc_score_encoder::doc_id`), so this
  half of the rule is a **declaration** gate, and the per-call-site audit for a
  method-syntax sentinel is still by hand.
- *A sentinel laundered through a wrapper.* If `a()` returns `-1` and `b()`
  returns `a()`'s value unchanged, only `b`'s own body is inspected.

## codec-suffix-literal

**Rule.** The per-field codec suffix is derived by
`index_writer::per_field_codec_suffix`, never spelled out. c14 shipped a
hardcoded `"Lucene90_0"` (its F-12).

**What the check does.** A string literal matching `LuceneNN_N` in non-test
code outside `index_writer.rs` fails.

**Blind spot.** A suffix assembled at runtime (`format!("{format}_{n}")`) is
indistinguishable from the sanctioned derivation and is not flagged.

## blocktree-infallible

**Rule.** Modules that consume `blocktree` use `try_seek_exact`/`try_next`/
`try_seek_ceil`/`try_current`, which surface a corrupt `.tim` block as an error
instead of degrading it to "no such term"/end-of-terms.

c39 completed this migration by hand -- marking the four infallible spellings
`#[deprecated]` and rebuilding -- and `blocktree.rs`'s method docs describe the
rule, but nothing ran it. Zero production call sites remain; the rule is a
regression guard, which is the only kind of gate that can be green on the day
it lands and still be worth having.

**Blind spot.** Scoped to files that name `blocktree`, because `fst.rs` has
same-named methods of its own. A wrapper type re-exporting the infallible
lookups from a module that never names `blocktree` is invisible. And the rule
checks `seek_exact`/`seek_ceil`/`current` but **not `next`**: `next` is
`Iterator`'s method name and matching it would fire on every iterator in the
workspace. `try_next`'s migration is therefore unguarded.

## doc-values-per-doc

**Rule.** `doc_values::numeric_value`/`binary_value` re-derive a column's
addressing from its entry on every call. `NumericReader`/`BinaryReader` derive
it once and are the sanctioned multi-lookup API. Calling the free function once
per document has already shipped twice (b13's
`soft_deletes::effective_live_docs`, c14's column merge).

**What the check does.** A call whose *ancestor chain* inside its enclosing
`fn` contains a `for`/`while` header or an iterator adaptor is a per-document
call. The ten that exist today are a **burn-down list** keyed by
`(file, enclosing fn)` in `DV_LOOP_BURNDOWN` -- the same shape as
`docs/arithmetic-gate.md`'s `TODO(arith-audit)` markers, for the same reason:
the debt has to be visible and it has to be able only to shrink. A new site
fails the gate; a migrated one fails it too, asking for the count to come down
in the same change.

Current burn-down (10 sites, 9 functions):

| file | fn | sites |
|---|---|---|
| `lucene-index/src/check_index.rs` | `check_doc_values` | 2 |
| `lucene-index/src/check_index.rs` | `doc_values_presence` | 1 |
| `lucene-index/src/check_index.rs` | `sort_key_values` | 1 |
| `lucene-index/src/merge.rs` | `merge_binary_doc_values` | 1 |
| `lucene-search/src/doc_value_query.rs` | `search_numeric_range` | 1 |
| `lucene-search/src/doc_value_query.rs` | `search_numeric_range_with_skip_index` | 1 |
| `lucene-search/src/doc_value_query.rs` | `sort_by_numeric_doc_value` | 1 |
| `lucene-search/src/doc_value_query.rs` | `sort_top_n_by_numeric_doc_value` | 1 |
| `lucene-search/src/facets.rs` | `count_single_valued` | 1 |

**Blind spots.** A per-document call hidden behind a helper the loop calls is
not seen -- the ancestor walk stops at the `fn` boundary. Indentation is the
nesting signal, so a `rustfmt`-illegal layout would confuse it (`cargo fmt
--check` runs first in the gate, which is what makes this safe).

**Why not `clippy::disallowed_methods`.** It fires on every call, including the
single-document lookups that are the API's whole point, and on all 66 sites in
test code. That is 60-plus `#[allow]`s whose proofs say nothing -- the failure
mode `docs/arithmetic-gate.md` names for a lint adopted too widely.

## parity ::item

`docs/parity.md` rows carry a Rust column like
``lucene-codecs/src/norms.rs::write_fields``. `check-parity.py` validated the
file path and **deliberately not** the `::item` suffix -- which is how c37's
Tier-2 review found `parity.md` describing two *deleted* functions in the
present tense. Since c41 the suffix is validated: every identifier a row names
must be defined (or re-exported) in the file the row points at. Two rows were
wrong when the check first ran.

**Blind spots.** Textual, not resolved: a name that exists somewhere in the
file satisfies it, and the check says nothing about whether the row's *prose*
is still true. Identifiers named in the status column are not checked at all.

## ledger-single-list

**Rule.** `docs/sweep/m2/LEDGER.md` contains no `- [ ]`.

The ledger has one reconciled list to plan from ("Open work, prioritised") and,
below it, a frozen archive of where each finding was first raised. Batches
repeatedly closed the prioritised entry and left its archive twin open. That is
the drift `c34-ledger-reconcile` was run to remove -- and **16 duplicate open
boxes survived that reconciliation and misled six later batches**, which is
what makes prose an inadequate enforcement mechanism for this.

The invariant is the smallest one that makes the failure impossible to express.
Every archive entry is:

| marker | meaning |
|---|---|
| `- [x]` | closed; the entry names the batch **and the evidence in the tree** |
| `- [~]` | obsolete; the premise stopped being true |
| `- [->]` | still open, and tracked as a numbered open-work item, which it names |

The prioritised list itself is numbered rather than checkboxed, so it is
unaffected.

**Blind spots.** It cannot tell whether a `- [x]` is *true* (c31's report
claimed a conversion it never made), nor whether a `- [->]` points at the right
item, nor whether a prioritised entry's prose still describes the tree. Those
stay human, and the ledger's own preamble names the two habits that catch them:
verify against the tree rather than a batch report, and treat a recorded
blocker as a claim with an expiry date.

## rustdoc

`rustdoc::broken_intra_doc_links` and its neighbours are warn-by-default and
reported by none of `cargo fmt`, `clippy`, `test` or `llvm-cov`. c4 shipped a
broken link through a fully green gate; c22 recorded the pass as "blocked on
pre-existing broken links" and it stayed recorded for four batches.

The gate runs:

```
RUSTDOCFLAGS="-D warnings -A rustdoc::private_intra_doc_links" \
  cargo doc --workspace --no-deps --document-private-items
```

`--document-private-items` is deliberate: this port's wire-format knowledge
lives in the doc comments of private decoders, and a link that only breaks
there is still a broken link.

`private_intra_doc_links` is **allowed**, with a count: 166 sites, essentially
all of them a public module doc pointing at the private helper that implements
what it is describing. That is the documentation working. Denying it would mean
deleting 166 useful links to satisfy a lint about rendered HTML.

**A rustdoc trap worth knowing.** An outer `///` doc on a `mod` declaration
makes rustdoc resolve *the whole merged doc* -- including the module file's own
`//!` lines -- in the **crate-root** scope, so every link the module writes to
its own items silently breaks and the diagnostic carries no file or line.
`lucene-codecs`'s `direct_reader`, `for_util` and `lz4` had this; the fix is to
keep the rationale in the module's own `//!` header, and `lib.rs` now says so.

**Reaching for a link rather than backticks.** An intra-doc link to a *private*
item in another module of the same crate does not resolve, even under
`--document-private-items` -- rustdoc reports "no item named X in module Y".
The temptation is to demote the link to plain backticks, which makes the gate
green by removing the thing it checks. Prefer widening the item to
`pub(crate)`: the link then resolves, a later rename breaks the build, and the
only cost is a `private_intra_doc_links` warning this gate already allows. c41
did that for `postings::LEVEL1_FACTOR`, `collector::rank_order`,
`multi_segment::global_term_stats` and `merge::merge_points`.

**Blind spots.** This is the big one: **rustdoc only checks symbols inside
`[...]`.** A symbol named in plain backticks -- which is most of them in this
tree, and all of them in `.md` files -- is invisible to it. That gap is what
the `parity ::item` rule covers for `docs/parity.md`, and what nothing covers
for `PLAN.md` or for prose in Rust comments. When a diff removes a `fn` or a
`struct`, grep `crates/`, `docs/parity.md` and `PLAN.md` for its name by hand;
`docs/sweep/` is an archive and is deliberately exempt.

---

## Running them

All of them are in `scripts/gate.sh` and therefore in
`scripts/docker-test.sh gate`, `.githooks/pre-commit` and CI. Individually:

```
python3 scripts/check-port-invariants.py --verbose      # counts per rule
python3 scripts/check-port-invariants.py --only=sentinel-callers
python3 scripts/check-port-invariants.py --only=ledger-single-list
python3 scripts/check-parity.py
RUSTDOCFLAGS="-D warnings -A rustdoc::private_intra_doc_links" \
  cargo doc --workspace --no-deps --document-private-items
```

## Each of these has been seen to fail

A gate nobody has watched fail is a gate nobody should trust -- this sweep
found three checks that could not fail and one that reported "pass" over a
segment it had never opened. Every rule above was verified by introducing the
defect it targets, watching it fire, and reverting; `c41-gates-and-record.md`
records the exact edit and the exact message for each.
