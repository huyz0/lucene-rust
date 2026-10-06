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
| [`parity ::item`](#parity-item) | `check-parity.py` | a ledger row (`docs/parity/*.md`) naming a Rust item its own file does not define, or a backticked `scripts/..`/`tools/..` path in its Status column that does not exist | other prose outside a row's Rust column; an item that exists but no longer does what the row says; a tool path written without backticks or outside `scripts/`/`tools/` |
| [`parity-layout`](#parity-layout) | `check-parity.py` | an area file the index does not link; a relative link to a missing file; a stated row count that is wrong; a row over 2,000 characters, a file over 80 KB, a ledger over 400 KB; a `## ` heading in two files | history written *within* budget; a row filed in the wrong area; a stale fact; a link to a file that exists but no longer says what the link claims |
| [`ledger-single-list`](#ledger-single-list) | `check-port-invariants.py` | an unticked `- [ ]` anywhere in `docs/sweep/m2/LEDGER.md` | whether a `- [x]` is *true*, or whether a `- [->]` names the right item |
| [`block-guard`](#block-guard) | `check-port-invariants.py` | a `lucene-index` fn setting `pending_has_blocks`/`dwpt.has_blocks = true` with no earlier `check_block(` call in the same fn | a guard that is called but whose result is ignored, or that sits on a branch the flag's line does not follow; a block flag spelled any other way |
| [`alloc-from-doc`](#alloc-from-doc) | `check-port-invariants.py` | an allocation size (`vec![_; n]`, `with_capacity(n)`, `.resize(n, ..)`, `FixedBitSet::new(n)`) mentioning a name its fn bound from a doc list's `.last()`/`.first()`/`.max()`, with no `max_doc` in the size and no `// ALLOC:` proof | a doc id reaching the size through a struct field, a parameter or another fn; a source not spelled on a `*doc*` name (`ids.last()`); a `max_doc` in the size that does not actually bound it |
| [`occur-guard`](#occur-guard) | `check-port-invariants.py` | one condition testing `R.must.is_empty()` and `R.must_not.is_empty()` on a receiver `R` without naming `R.filter` | a guard split across statements or helper fns; a test on `should`/`filter` alone that forgets an occur; a `filter` named but treated wrongly |
| [`table-fixed-len`](#table-fixed-len) | `check-port-invariants.py` | a non-test `static`/`const` in `lucene-analysis` typed as a slice literal (`&[T] = &[`) | a fixed-length type whose `N` was itself copied from the short transcription; a table built at runtime (`vec!`, `LazyLock`); a dropped or reordered entry in a table nothing indexes by position; every other crate |
| [`alloc-from-parse`](#alloc-from-parse) | `check-port-invariants.py` | an allocation size (`vec![_; n]`, `with_capacity(n)`, `.resize(n, ..)`) mentioning a name its fn bound from a text parse (`parse_int(`, `parse_num(`, `.parse()`) or containing one, with no `min(` in the size and no `// ALLOC:` proof | a count reaching the size through a struct field, a parameter or another fn; a count decoded from bytes (`read_vint`) rather than parsed from text; a `min(` that does not actually cap it; a loop that pushes once per announced entry without reading one |
| [rustdoc links](#rustdoc) | `cargo doc` | a `[`link`]` that resolves to nothing | a symbol named in *plain backticks*, which is most of them |
| [coverage objects](#coverage-objects) | `scripts/coverage.sh` | lucene-ffi's never-loaded cdylib standing in for the executed copy of a `#[no_mangle]` function in the line report | the same first-copy-wins rule between two *executed* copies; a file whose lines split between a crate's unit-test and integration-test builds, which the summary under-reports |

Between them these six rules cover **the indexing row** of the
arithmetic gate's table (`FixedBitSet` only) and **the two hand-checked rules**
at the end of that document. Apart from `alloc-from-doc`'s and
`alloc-from-parse`'s shapes (a size taken from a decoded document id, or from
a count parsed out of a text header) they do not cover slicing or allocation
sizing: that is still a hand audit, still step 2 of the three-part module audit, and
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
   a new sentinel cannot arrive unannounced. There are nine today.
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

Current burn-down (7 sites, 6 functions):

| file | fn | sites |
|---|---|---|
| `lucene-index/src/check_index.rs` | `check_doc_values` | 2 |
| `lucene-index/src/check_index.rs` | `doc_values_presence` | 1 |
| `lucene-index/src/check_index.rs` | `sort_key_values` | 1 |
| `lucene-search/src/doc_value_query.rs` | `search_numeric_range_with_skip_index` | 1 |
| `lucene-search/src/doc_value_query.rs` | `sort_by_numeric_doc_value` | 1 |
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

Ledger rows (`docs/parity/*.md`) carry a Rust column like
``lucene-codecs/src/norms.rs::write_fields``. `check-parity.py` validated the
file path and **deliberately not** the `::item` suffix -- which is how c37's
Tier-2 review found `parity.md` describing two *deleted* functions in the
present tense. Since c41 the suffix is validated: every identifier a row names
must be defined (or re-exported) in the file the row points at. Two rows were
wrong when the check first ran.

**Blind spots.** Textual, not resolved: a name that exists somewhere in the
file satisfies it, and the check says nothing about whether the row's *prose*
is still true. Identifiers named in the status column are not checked at all.

The Status column's backticked `scripts/..` and `tools/..` paths (generators
and verifiers cited as evidence) are checked too, since the M11 part 2 review
found the Snowball rows citing `tools/snowball_utf16_tables.py`, which never
existed: `scripts/` resolves against the repository, `tools/` against the
repository or a crate the row's Rust column names (`crates/<crate>/tools/`).
**Seen to fail** on that row: `docs/parity/analysis-common.md:18: tool or
script does not exist: tools/snowball_utf16_tables.py`. Blind to a path
outside those two directories, one not in backticks, or one that exists but
no longer does what the row says.

## parity-layout

**Rule.** The parity ledger is `docs/parity.md` (an index) plus one table per
area in `docs/parity/`. `check-parity.py` fails when the index does not link
an area file or states its row count wrongly; when any relative Markdown link
in the ledger points at a file that does not exist; when a table row is over
2,000 characters, a file over 80 KB or the whole ledger over 400 KB; and when
one `## ` heading appears in two files.

The ledger was one 1.25 MB file whose largest row was 39,508 characters: every
batch appended its own history (dates, "previously", test counts, superseded
gaps) to the row it touched, so the row's current state had to be dug out of
the last of several contradicting paragraphs. Splitting it removed the
history; the budgets keep it from growing back, and the index checks keep the
split navigable (an orphaned file is a file nobody finds).

Seen to fail, then reverted: an unlinked `docs/parity/orphan.md` ("the index
does not link"); `[Index](../parity-old.md)` in `store.md` ("link target does
not exist"); a 2,275-character grouping row ("over the 2000 budget"); 85 KB of
padding in `store.md` (file and ledger budgets); `## Supported` in two files
("appears in several files"); an extra grouping row ("the area table says ...
has 11 rows, it has 12").

**Blind spots.** Size is a proxy: a row can carry history in 1,900 characters,
and only review catches a "previously" or a superseded gap that the next
sentence contradicts. Nothing checks that a row sits in the right area file,
that its prose is still true, or that a link's target still says what the
link claims. Links with an absolute URL or only an anchor are not checked.

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

## port-inventory

`scripts/check-port-inventory.py`, over `docs/inventory/lucene-core.tsv`:
every top-level class of Lucene 10.5.0's `lucene-core` jar (1,196, the
Java 21 multi-release variants included), one status each -- `ported` with a
Rust file and symbol, `partial` with the milestone that closes its gap,
`not-needed` with a reason, `todo:M<n>`/`deferred:M<n>`. `parity.md` records
what someone chose to write down; this records what nobody did, which is what
"fully ported" has to be measured against. `--milestone M7` lists what M7
still owes; `--summary` counts by status and package.

**Seen to fail**: a deleted row (`index/IndexWriter: in the jar, not in
lucene-core.tsv`), a `ported` row naming a symbol its file lacks, and a
`partial` row without its milestone tag.

**`--module spatial-extras`** (M9 T9.5, `docs/inventory/lucene-spatial-extras.tsv`,
the 65 top-level classes of `lucene-spatial-extras`) was seen to fail the
same three ways before it joined `gate.sh` and CI: a deleted row
(`spatial/vector/PointVectorStrategy: in the jar, not in
lucene-spatial-extras.tsv`), a renamed symbol (`... spatial4j.rs has no
`Geo3dBinaryCodecX``), and an extra row (`spatial/Bogus: in
lucene-spatial-extras.tsv, not in the jar`); `--milestone M9` lists what T9.5
still owes. It cannot see the third-party code the module is built on
(Spatial4j, S2): those subsets are not Lucene classes, so `docs/parity.md`
records them, and their correctness rests on `spatial4j_fixtures.rs` alone.

**`--module join`, `--module grouping`, `--module queries`** (M10 T10.0,
`docs/inventory/lucene-{join,grouping,queries}.tsv`: the 29, 30 and 157
top-level classes of those jars) were each seen to fail before they joined
`gate.sh` and CI: a deleted row (`search/join/BitSetProducer: in the jar, not
in lucene-join.tsv`, `search/grouping/AllGroupHeadsCollectorManager: ...`,
`queries/CommonTermsQuery: ...`), an extra row (`search/Bogus: in
lucene-<module>.tsv, not in the jar`, all three), a renamed symbol
(`queries/spans/SpanScorer: crates/lucene-search/src/exec/span.rs has no
`span_doc_scoresX``) and, with `$JARS`, `$HOME` and the proxy pointing
nowhere, `no lucene-join jar (...); --require-jar makes that a failure`.
Every class M10 has not ported yet is `todo:M10`, so `--milestone M10` lists
what the milestone still owes. The `queries` rows marked `ported` are the span
queries M2/M7 ported as algorithms rather than classes (`SpanTermQuery`,
`SpanNearQuery`, `SpanOrQuery` and the iterators under them); the gate checks
that the cited function exists, not that it covers the whole Java class.

**`--module analysis-common`** (M11 T11.0,
`docs/inventory/lucene-analysis-common.tsv`: the 498 top-level classes of
`lucene-analysis-common`, the 33 `org/tartarus/snowball/` classes of the
Snowball runtime and its generated stemmers included -- the jar ships them, so
the checker now reads classes outside `org/apache/lucene/` too and keeps their
full path) was seen to fail before it joined `gate.sh` and CI: a deleted row
(`analysis/core/WhitespaceTokenizer: in the jar, not in
lucene-analysis-common.tsv`), a deleted Snowball row
(`org/tartarus/snowball/Among: ...`), an extra row (`analysis/Bogus: in
lucene-analysis-common.tsv, not in the jar`), a renamed symbol
(`keyword_tokenizer.rs has no `KeywordTokenizerX``) and a missing jar. Every
class M11 has not ported is `todo:M11` tagged with its task (T11.2 JFlex,
T11.3 Snowball, T11.4 Hunspell, T11.5 synonyms, T11.6 per package, T11.7
factories), so `--milestone M11` lists what the milestone still owes.

**Blind spots.** It checks that a cited symbol *exists*, not that it does
what the Java class does -- the classification is a judgement, made per class
against the Java source and recorded in the row, and a wrong `ported` passes.
It covers `lucene-core` only; other modules get their own file as their
milestones start. Without a jar (offline, no Gradle cache) the membership
check is skipped and says so; the per-row checks still run.

**`--require-jar`** (T9.4 review): `gate.sh` and CI pass it, so a jar that
cannot be found fails the check instead of reporting "ok" over a membership
check that never ran. Before it, the script also ignored `$JARS` (it
hard-coded `fixtures/.jars`), so inside the container -- jars baked into
`/opt/lucene-jars`, no network -- every module's membership check was
silently skipped. **Seen to fail**: with `$JARS` pointing nowhere, no Gradle
cache and the proxy unreachable, `no lucene-spatial3d jar (looked in $JARS
or fixtures/.jars, the Gradle cache and Maven Central); --require-jar makes
that a failure`; with `$JARS` holding only the spatial3d jar (no cache, no
network) and a row deleted, `spatial3d/geom/GeoWorld: in the jar, not in
lucene-spatial3d.tsv` -- the old script would have skipped that. It cannot
tell a *wrong* jar from the right one: whatever `lucene_resolve_jar` returns
for `<module>-10.5.0.jar` is trusted, so a corrupt or substituted jar file
under that name passes as long as it opens as a zip.

## block integrity (M10 T10.1)

**`VerifyJoin`** (`scripts/verify-write-path.sh`, `write_block_join_fixture`):
real Lucene runs `CheckIndex` and `CheckJoinIndex` over two Rust-written block
indexes (one index-sorted), finds every live block through the join queries,
appends with its own `IndexWriter` and force-merges. **Seen to fail** with the
flush's parent-key wrap (`key_of_parent` in `sort_buffer`) removed: `CheckJoinIndex:
Parent doc 31 of segment _cz ... is live but has a deleted child document 30`.
Lucene's own `CheckIndex` passed that index: `testSort` walks only the
parents, so a shredded block is invisible to it -- which is why the verifier
does not stop at `CheckIndex`.

**`block_join_merge_stress.rs`** (24 seeded streams under a merge-happy
policy, then `check_join_index` and every block read back): **seen to fail**
with the merge's parent-key wrap removed (`seed 100 sorted=true: Parent doc
111 of segment _4v is live but has a deleted child document 96`). Blind spot,
shared with `op-stream-fuzz.sh`'s block lines: a block kept contiguous but
reordered *within* itself would pass the join checks; the stress test's
in-order child ids are what catch that.

## block-guard

`check-port-invariants.py --only=block-guard`. Java's
`DocumentsWriterPerThread.updateDocuments` refuses a block (two or more
documents) in a sorted index without a parent field before it buffers
anything. The T10.1 review found two writer paths (the native
`add_documents_with_delete` and the concurrent writer's `add_entries`) that
buffered such a block, after which every flush failed in `sort_buffer` and the
good documents could only be dropped by a rollback. The rule: in
`crates/lucene-index/src`, outside tests, every fn that sets
`pending_has_blocks = true` or `dwpt.has_blocks = true` must call
`check_block(` on an earlier line of the same fn (brace-counted spans, as the
other rules). **Seen to fail** with the guard removed from `add_entries`:
`concurrent_writer.rs:843: `add_entries` marks its buffer as holding document
blocks without calling `check_block(..)` first`. Blind to: a call whose
`Result` is discarded (`let _ =`), a guard on a sibling branch, and a flag set
through any other name (a helper taking `&mut bool`).

## alloc-from-doc

`check-port-invariants.py --only=alloc-from-doc`. The M10 T10.3-T10.4 review
found `reader::segment::SparseDocs::new` sizing its bit words and rank table
from the **last decoded document** of a sparse field's `IndexedDISI` instead
of the segment's `maxDoc`: eight corrupt bytes (a SPARSE block numbered
`0x7FFF`) named document `0x7FFF_FFFE` in a ten-document segment, and the
reader cached a ~400 MB allocation for its life -- or, when the allocation
failed, aborted the process and the JVM with it. The fix sizes from `max_doc`
and refuses a document at or past it, or out of order, as it decodes
(`indexed_disi::decode_doc_ids_below`). The rule: outside tests, in any fn, an
allocation size must not mention a name bound (directly or through other
`let`s of the same fn) from a `*doc*` list's `.last()`/`.first()`/`.max()`,
unless the size expression names `max_doc` or an `// ALLOC:` comment within
the six lines above states the bound that makes it sound
(`lucene_util::doc_id_sort`'s radix histogram carries the one such proof).
**Seen to fail** on the unfixed `SparseDocs::new`:
`crates/lucene-search/src/reader/segment.rs:64: `new` sizes an allocation from
a decoded document id (`end.div_ceil(64)`)` and, through `words`, line 70's
`Vec::with_capacity(words.len())`. Blind to: a document id that reaches the
size through a struct field, a parameter or a helper fn; a list not named
`*doc*`; and a `max_doc` in the size that does not bound it (`max_doc.max(n)`).

## toplevel-whole-reader

`check-port-invariants.py --only=toplevel-whole-reader`. The M10 T10.5 review
found `exec/function.rs`'s `context()` and `rewritten()` falling back to
`TopLevel::single(*ctx)` -- the segment being scored as the whole index --
whenever the leaf had no statistics pass, and the sorted, counted, aggregated
and `terminate_after` entry points ran that pass only when they scored. A
`scale`, `docfreq`, `idf`, `maxdoc`, `numdocs`, `joindf` or
`IndexReaderFunctions` source then read one segment's statistics: wrong hits
on any multi-segment index, with nothing to say so. Every entry point now
prepares the query's function queries over all its segments
(`multi_segment::global_function_stats`, or the full statistics when it
scores), and a leaf reached unprepared is an `IllegalState` error. The rule:
outside tests, no `TopLevel::single(` anywhere, and no `TopLevel { .. }`
literal outside `function/mod.rs` (where `prepare_functions` and
`TopLevel::of_searcher` build it from every segment). **Seen to fail** on the
unfixed `exec/function.rs`: `crates/lucene-search/src/exec/function.rs:47:
`TopLevel::single(` builds a function query's top-level reader from one
segment.` (and line 64), and on a probe `TopLevel { leaves: vec![c] }` in
`exec/function.rs`: `a `TopLevel` built outside `function/mod.rs``. Blind
to: a `TopLevel` built inside `function/mod.rs` from fewer segments than the
search covers (a slice of the segments passed to `prepare_functions`); an
entry point that hands its leaves a `GlobalStats` prepared for a different
query (the leaf then errors, it does not read wrong statistics); and a
function query reached through a path that builds leaf contexts without a
statistics pass -- that is a runtime error, not a gate failure.

## occur-guard

`check-port-invariants.py --only=occur-guard`. The M10 T10.6 review found
`BooleanQuery::rewrite`'s rule 6 flattening a nested disjunction out of
`should` on `inner.must.is_empty() && inner.must_not.is_empty()`. Java's
`isPureDisjunction()` is "every clause is `SHOULD`": the guard named three of
the four occurs, so `[(body:a #body:b), body:c]` flattened to `[body:a,
body:c]` and dropped `#body:b`. `FILTER` arrived in this port after the other
three, and a guard written before it compiles unchanged after it. The fix is
`BooleanQuery::is_pure_disjunction`, Java's predicate by name. The rule:
outside tests, a condition -- the code between two of `{`, `}`, `;`, `=>` --
that tests `R.must.is_empty()` and `R.must_not.is_empty()` on one receiver
`R` (`inner.`, `self.`, or none for locals) also names `R.filter`. Five such
conditions exist today, all naming it. **Seen to fail** on the unfixed guard:
`crates/lucene-search/src/query.rs:1275: a guard tests `inner.must` and
`inner.must_not` but not `inner.filter``, and on `is_pure_disjunction` with
its `filter` line deleted. Blind to: a guard whose tests are split across
statements, `let`s or helper fns; one testing `should` or `filter` without
`must` and `must_not` (a three-occur test of another shape); `len()` or
iterator tests instead of `is_empty()`; and a `filter` that is named but
handled wrongly -- the rule proves the fourth occur was *thought about*, not
that the answer is right.

## table-fixed-len

`check-port-invariants.py --only=table-fixed-len`. The M11 part 1 review
found `cjk::KANA_COMBINE_HALF_VOICED` typed `&[u8]` with 86 entries where
Java's `byte[]` has 88: `combine` indexes it by `prev - 0x30A6` for `prev`
up to U+30FD, so `ーﾟ` panicked in `CJKWidthFilter`, `CJKWidthCharFilter`
and `CJKAnalyzer` (Java: `30fc 309a`). A slice type accepts any length; an
array type makes the compiler count. The rule: outside tests, no `static` or
`const` in `crates/lucene-analysis/src` is a slice literal (`: &[..] = &[`,
`: &'static [..] = &[`). Indexed tables spell `N` from the range Java
indexes them by (`[u8; 0x30FD - 0x30A6 + 1]`), so the count is checked
against Java's arithmetic, not against the transcription. A reference to an
array (`&'static [T; N] = &[`, the Snowball compiler's tables) is an array
type: the rule looks for a `;` at the outer bracket's depth, so `[[u8; 4]]`
stays a slice. **Seen to fail** on
the original `&[u8]` declaration (`crates/lucene-analysis/src/cjk/mod.rs:284:
table `KANA_COMBINE_HALF_VOICED` is a slice literal`) and on
`PROPER_NOUNS: &'static [&str]`, and on the generated Snowball `STEMMERS:
&[(&str, fn(..) -> bool)]` while passing its `&'static [Among<Context>; N]`
tables; with the array type, the 86-entry
transcription is `error[E0308]: mismatched types`. Blind to: an `N` copied
from the short transcription rather than derived (only list tables do this,
and every one was counted against the Java source or resource when the rule
landed); a table built at runtime; a dropped or reordered entry in a list
nothing indexes by position (a stop list); other crates.

## alloc-from-parse

`check-port-invariants.py --only=alloc-from-parse`. The M11 part 2 review
found Hunspell's `AF` and `AM` parsers (`hunspell/dictionary.rs`) pre-sizing
their alias tables as `vec![Vec::new(); count.max(0) as usize]` with `count`
read off the `.aff` header: `AF 2000000000` asked for 48 GB and aborted the
process (Java's `new String[count]` throws `OutOfMemoryError`, which no caller
catches either). A count a text format announces is a promise about lines
still to come: the port reserves at most 1,024 slots (`Aliases::new`) and
grows as the lines arrive, refusing a line past the count as Java does. The
rule: outside tests, in any fn, an allocation size must not mention a name
bound (directly or through other `let`s of the same fn) from `parse_int(`,
`parse_num(`, `.parse::<..>()` or `.parse()`, nor contain such a call, unless
the size names `min(` or an `// ALLOC:` comment within the six lines above
states the bound. **Seen to fail** on the unfixed parsers:
`crates/lucene-analysis/src/hunspell/dictionary.rs:1400: `parse_alias` sizes
an allocation from a count parsed out of its input (`count.max(0) as
usize`)` and the same at line 1435 for `parse_morph_alias`. Blind to: a count
that reaches the size through a struct field, a parameter or another fn
(`Aliases::new(count)` is checked only because it caps with `min(`); counts
decoded from binary (`read_vint`, which the arithmetic gate and
`alloc-from-doc` cover in part); a `min(` against a bound that is itself too
large; a loop that pushes once per announced entry without consuming input.

## write-path verifiers of the geo modules (M9)

`scripts/verify-write-path.sh` runs a Java verifier over an index this
port's `IndexWriter` wrote; for the geo modules each verifier answers every
recorded question over the Rust index and over Lucene's own (the committed
fixture), in one JVM, and requires the answers equal.

**`VerifySpatialExtras`** (T9.5, `write_spatial_strategies_fixture`): seen to
fail with the prefix-tree strategies' token streams dropping every legacy
leaf cell (`term.last() == Some(&b'+')`, skipped in `field_of`):
`differs: q rgh Contains g:ENVELOPE(175.31..., -152.41..., ...)  lucene's
index: C 9 ...  rust's index: C 0`. **`VerifyGeo3D`** (T9.4) was seen to
fail the same way in T9.5: with `Geo3DPoint.encode_dimension` halving every
encoded value divisible by three, `differs: dist p 9.27 179.90 5240816.28 --
lucene's index: C 878 ...  rust's index: C 568 ...`.

**Blind spots, observed.** A field written wrongly in a way no recorded
answer reflects passes. Measured: `encode_dimension` adding one unit to every
encoded value divisible by 97 left all 260 of `VerifyGeo3D`'s answers equal --
while `every_geo3d_point_is_indexed_as_lucene_indexes_it` failed on it. The
byte-level Rust tests (`every_geo3d_point_is_indexed_as_lucene_indexes_it`,
`every_spatial_strategy_makes_the_fields_lucene_makes`) are what pin the
encoding; the verifiers pin that Lucene reads the result and answers alike.

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

## release-profile

`cargo test --release -p lucene-search`, in `scripts/gate.sh` and CI's gate
job (x64 and arm64). Every other test step runs a debug build, which is not
what the plugin ships. The optimiser may reorder a floating-point addition,
and nothing preserves a `NaN`'s sign or payload across that: x86 makes
`inf + -inf` the negative default `NaN`, aarch64 the positive one, and with
two `NaN` operands keeps whichever comes first. The metric aggregations' sum
came out `0xfff8...` under `--release` where Java's fixture records
`Double.NaN`, and `metric_aggs_fixtures` failed in release only.

**Seen to fail** on the commit before `f571c71`: `metric_aggs_fixtures`, three
slices of field `d` (`got 38429:fff8000000000000:...`,
`Lucene 38429:7ff8000000000000:...`).

**Blind spots.** Only `lucene-search`: `lucene-codecs`' and `lucene-ffi`'s
tests, which also compare floats bit for bit, still run in debug only, as
does every Java-side check. A difference that needs a CPU other than x64 or
arm64, or a toolchain other than the pinned one, is not reached. The step
adds `lucene-search`'s release build to the gate's time (about 1m40 on a
warm dependency cache).

## coverage-objects

`scripts/coverage.sh`, which `scripts/gate.sh` and CI's coverage step call in
place of `cargo llvm-cov`: it runs the tests (`cargo llvm-cov --no-report`),
deletes `liblucene_ffi*.so` from `llvm-cov-target/`, then reports (`cargo
llvm-cov report`). cargo-llvm-cov passes every executable file under its
target directory to `llvm-cov` as an `-object`, the cdylib included, though no
test loads it. Each `#[no_mangle]` entry point is in it under the same
unmangled name as the executed copy, and `llvm-cov` keeps the first record it
loads for a name. When that was the cdylib's, the report showed a function
nobody ran: lucene-ffi's `directory.rs` 91.9%, `results.rs` 91.8%,
`results_sorted.rs` 90.7%, `results_scored.rs` 94.8%, `ffm_bridge.rs` 94.2%,
`explain.rs` 94.7%, crate 96.0% -- and 100, 100, 100, 100, 99.9, 96.3 and
98.2% without it.

**Seen to fail**: with `results_sorted.rs`'s two buffer tests un-`#[test]`ed,
`scripts/coverage.sh -p lucene-ffi -- --summary-only` reports it at 88.0%
(from 100%) and CI's per-file filter lists it; reverted. Without the deletion
the same file reads 90.7% with every test in place -- the phantom this removes.

**Blind spots.** First-record-wins also applies between two *executed* copies:
lucene-ffi's unit-test binary and `tests/resource_bounds.rs` (which links the
rlib) carry the functions it calls under different function hashes, and the
report shows whichever `llvm-cov` loads first -- the order is the target
directory's, not sorted. And `llvm-cov`'s per-file *summary* takes, for each
function, its best single copy rather than the union of copies, so a file
whose lines are split between a crate's unit-test build and its
integration-test build is under-reported: `lucene-codecs`'
`backward_codecs/postings.rs` reads 94.5% (20 lines missed) while
`--show-missing-lines`, which is the union, names 5 (98.6%). The gate's
`--fail-under-lines` uses the summary, so it errs low, never high.

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

## The Snowball stemmers' gates (M11)

Not in `scripts/gate.sh` (they download pinned inputs and need a C compiler);
CI job `snowball vocabularies`.

| gate | where | catches | seen to fail by | blind to |
|---|---|---|---|---|
| generated stemmers | `scripts/check-snowball-generated.sh` | a file under `lucene-analysis/src/snowball/algorithms` that is not byte for byte `tools/gen_snowball.sh`'s output over the Snowball sources at `34f3612e` (a hand edit, a generator change nobody re-ran); Snowball sources or Lucene's sources jar other than the pinned ones (SHA-256 over the source tree file by file, and of the jar); the compiler's Java backend no longer reproducing Lucene's stemmers | one space inserted into `english.rs`'s first `slice_from` (`189c189 < env.slice_from(&[89]); > env.slice_from(&[ 89]);`) | whether the generator's four rewrites are *right* (the fixtures and the vocabularies are); the runtime, `program.rs`, which is written by hand |
| snowball vocabularies | `scripts/check-snowball-vocabulary.sh` -> `tests/snowball_vocabulary.rs` (skipped without `SNOWBALL_DATA`) | a stem of any of the 11.3 million snowball-data `f08c4d63` words that differs from its `output.txt` (where Lucene 10.5.0's own stemmers agree word for word); a moved data file (SHA-256 per file) | one Danish `output.txt` line changed (`Danish: 1 of 23830 differ, first ["adami -> adami (snowball-data: adamix)"]`) | words outside the vocabularies (the committed fixtures cover the synthetic ones); the filter around the stemmer (`SnowballFilter`, which the committed fixtures run) |

## The OpenSearch plugin's gates (M2)

These are not in `scripts/gate.sh` — they need a JVM, Docker, or a nightly
toolchain — but they are CI jobs (`opensearch`, `fuzz`) and each has been seen
to fail by planting the defect it targets.

| gate | where | catches | seen to fail by | blind to |
|---|---|---|---|---|
| differential self test | `gradle -p opensearch-plugin check` (`NativeSelfTest`) | a native hit, score (1e-5), count or threshold-count that differs from Lucene's `IndexSearcher` on the same NRT reader; a JNI error path that throws or crashes instead of returning a status; a native reader outliving its Java reader | never sending live docs (1,218 failures); dropping `minimumNumberShouldMatch` from the blob (44); never closing native readers (31); reporting "≥" *at* the count threshold instead of past it | shapes its random generator does not produce (phrases, multi-term queries — which the encoder refuses anyway); a divergence smaller than 1e-5 |
| node matrix | `scripts/verify-opensearch.sh` (`verify_opensearch.py`) | a REST response (hits, scores, totals, max score, highlights, aggs) that differs with the plugin on; a query that runs native when it should fall back, or the reverse, or for the wrong reason | routing drift in both directions during development: `boosted match` running native while the matrix still expected `query_BoostQuery`, and `size 0` answered by the request cache (neither route counted) until the matrix set `request_cache=false` | request shapes not in its matrix; multi-node clusters |
| reader release | the same script, `lifecycle` | a native reader still open after its Java reader closed, and a merged-away file still mapped | never closing native readers: the open-reader count at round 1, the deleted-file mapping at round 2 | a leak of **compound** segments by mapping, because the native reader copies `.cfs` segments rather than mapping them — only the reader count sees those, which is why both checks exist |
| YAML suites vs stock | `verify-opensearch.sh --yaml` | any test of OpenSearch's own REST suites that fails differently with the plugin; any native error; a run where nothing ran native | a `postings_format` gap (completion fields) surfaced as a native error — the check that fired was the native-error count, not a test failure, because fallback kept the answer right | suites not in `YAML_SUITES`; failures that also happen on the stock node (4 `_source`-filtering warning-header tests), which it deliberately does not judge |
| derived engine tests | `scripts/check-derived-engine-tests.sh`, CI `opensearch engine` (before the tests run) | a derived file under `opensearch-plugin/src/engineTest` (`RustEngineTests`, `RustEngineTestCase`, `RustTestEngine` and the three server helpers) that is not byte for byte what `derive_engine_tests.py` writes from OpenSearch 3.8.0's sources: a hand edit (an `@Ignore` added, a test body changed), or a change to the script nobody re-ran; a moved `3.8.0` tag or a republished sources jar (each input has a pinned SHA-256) | a hand-added `@Ignore("hand edit")` on `RustEngineTests.testSegmentsWithUseCompoundFileFlag_true` (the diff named the line); a reason string changed in `SKIPPED` without re-deriving; one byte appended to `InternalEngineTests.java` (the checksum) | the hand-written files beside them (`RustIndexerFactoryTests`, `EngineTestAccess`); whether `SKIPPED` is *right* -- a test skipped with a true-sounding reason it does not deserve passes; OpenSearch tests outside `InternalEngineTests` (the script derives no others) |
| FFI fuzzing | `crates/lucene-ffi/fuzz/`, CI `fuzz` | a panic (even a caught one), ASan-visible memory error, or implausible result on arbitrary query blobs, `SegmentInfos` bytes, live-docs words and clause arrays | no finding in 28M+ runs across the four targets; the asserts themselves were checked by planting, in a scratch copy, a `panic!` on query tag 7 (`jvm_search` stopped on "a query blob panicked") and a doc ID offset by 100 (stopped on "a doc id outside the index"), each within seconds | the JNI shim (fuzzed through the C ABI it calls, not through a JVM); inputs past the seeds' reach in a 60-second CI run |
