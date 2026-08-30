# The arithmetic gate

A mechanical guard against the single most productive defect class this port
has found: **a length, count, offset or width read off disk used in
arithmetic, indexing, slicing or an allocation without a bound.**

## Why it exists

The failure is not a wrong answer, it is a dead process:

| shape | debug build | release build | through the FFI |
|---|---|---|---|
| `base + len_from_disk` overflows | **panic** | silent wrap → wrong bytes | panic or corruption |
| `x << width_from_disk` | **panic** | silent wrap | same |
| `buf[off_from_disk]` / `&buf[a..b]` | **panic** | **panic** | same |
| `Vec::with_capacity(len_from_disk)` | **abort** | **abort** | **dead JVM** — `catch_unwind` cannot intercept an allocation failure |

The M2 sweep found this class **by hand, nine separate times**: b1 (three
`read_string`-family allocations), b2 (five unvalidated bit widths), b4 (block
region lengths, terms-dict prefix), b6 (live-docs ghost bits), b7 (`.kdm`
`numLeaves`), c8 (four `debug_assert` sites), the main session (a fifth
`debug_assert` site and a tail-block slice), c15 (three overflowing length
additions and five disk-sized allocations).

The argument for a *mechanical* gate is c15's own experience: c15 ran a
deliberate audit of `postings.rs` for exactly this class, fixed eight sites —
and its Tier-2 review then found **eight more of the same class in the same
file** that the audit had walked past. Careful manual auditing demonstrably
does not catch them all.

## The gate

`clippy::arithmetic_side_effects`, denied per crate through
`[workspace.lints.clippy]` in the root `Cargo.toml` plus `[lints] workspace =
true` in the crate's own manifest. It is part of the existing gate — `cargo
clippy --workspace --all-targets -- -D warnings` — with no extra command to
remember and no separate CI job.

It covers the arithmetic and shift rows of the table above. It does **not**
cover indexing/slicing or allocation sizing; see [Lints considered and not
adopted](#lints-considered-and-not-adopted).

### Where it is on

| crate | state | remaining `TODO(arith-audit)` modules |
|---|---|---|
| `lucene-store` | **on, fully audited** | none |
| `lucene-codecs` | **on, fully audited** | none |
| `lucene-index` | **on, fully audited** | none |
| `lucene-util`, `lucene-analysis`, `lucene-search`, `lucene-core`, `lucene-ffi` | off | — |

All three crates it is on for are now fully audited; c30 closed the last
three modules (`index_writer`, `merge`, `merge_policy`).

The three crates it is on for are the ones that turn *bytes on disk* into
values. The others consume values those crates have already validated, or (in
`lucene-util`/`lucene-analysis`'s case) do not read files at all. Turning it on
there is a defensible future change, not a correctness gap today.

### How a crate is adopted incrementally

The deny is **crate-wide**, so a *new* module is gated from its first line and
nobody has to remember to opt it in. Modules that predate the gate carry a
one-line opt-out on their declaration in the crate's `lib.rs`:

```rust
#[allow(clippy::arithmetic_side_effects)] // TODO(arith-audit)
pub mod blocktree;
```

That marker is the burn-down list. To audit a module: delete its marker, run
`cargo clippy -p <crate> --all-targets -- -D warnings`, resolve everything it
reports by the rules below, and tick the module off in the table above.

Keeping the markers in `lib.rs` rather than at the top of each module file is
deliberate: the whole burn-down is visible in one place, and adopting the gate
does not touch 60 files owned by other in-flight work.

### A module is not audited when the lint goes quiet

The lint covers **arithmetic and shifts**. It does not cover indexing,
slicing or allocation — see [Lints considered and not
adopted](#lints-considered-and-not-adopted) for why those are not gated
mechanically. c25 proved the gap the hard way: its re-signed `.tvd` byte-flip
sweep found two defects in an already-lint-clean `term_vectors.rs`, a
`prefixLength` off disk slicing the previous term (panic) and a claimed
decompressed length sizing `vec![0u8; n]` (**SIGABRT**, ~1 PB).

So an audit has three parts, and clippy is only the first:

1. Resolve every lint site by the three rules above.
2. **Hand-check every `slice[i]` / `&slice[a..b]` whose index came from file
   contents, every `Vec::with_capacity(n)` / `vec![0; n]` / `resize(n, _)`
   whose `n` did, and every `copy_from_slice`/`copy_within` whose length did.**
   c27 found four aborts and one release-mode infinite loop this way, none of
   which the lint reported.
3. **Add a re-signed byte-flip sweep** for the files the module parses: flip
   each byte in turn, re-sign the codec footer so only semantic invariants can
   fire, and require a typed error or a clean decode — never a panic and never
   an abort. Record the rejection rate. Measured so far: `.fdm` 269/282,
   `.fdi` 169/198, `.tim`+`.tip`+`.tmd` 391/436, `.kdm` 270/378, `.dvm`+`.dvd`
   1 686/2 564, `.nvm` 85/99, `.fdx` 114/126, `.fdt` 624/904, `.tvd`+`.tvm`
   260/452, `.kdd` 204/438, `.tip` 44/99, `.dvd` 18/99, `.vem`+`.vex`
   1 656/3 536, `.vemf`+`.vec` 465/26 168 (`.vemf` alone 350/508), the FST
   fixture corpus 2 853/7 680 and a 4 KB built FST 4 183/32 456. A low rate is not
   automatically a gap — most `.kdd` and `.fdt` payload bytes are values, and
   flipping one yields a different but perfectly well-formed record. The bar
   is that **nothing panics, aborts, or reserves memory proportional to a
   number it just read.**

Two reproduction notes. An allocation abort may not reproduce on a machine
with Linux overcommit — a 498 GB mapping can succeed and the unfixed code then
fails on a later error instead. Run the unfixed case under
`( ulimit -v 4000000; cargo test ... )` so the allocation actually fails. And
build the fixture with **more than one chunk/block**: c27's first
`.fdm` sweep scored 211/282 against a single-chunk segment, because a
one-chunk segment gives the monotonic index arrays nothing to discriminate —
that rate was measuring the fixture, not the decoder.

## The rule for writing code under the gate

Every `+`, `-`, `*`, `/`, `%`, `<<`, `>>` and unary `-` on an integer must be
one of:

1. **`checked_*`, with the `None` reported as corruption.** The default for
   anything whose operands came off disk. Prefer folding the check and the
   arithmetic into one operation:

   ```rust
   let Some(footer_start) = total_len.checked_sub(FOOTER_LENGTH) else {
       return Err(corrupt(format!("misplaced codec footer: length={total_len}")));
   };
   ```

2. **`saturating_*` / `wrapping_*`, where that is the honest semantics.**
   `wrapping_add` for an accumulator that is porting a Java `int` that wraps
   (see `postings.rs`'s position/offset accumulators). `saturating_add` for a
   counter whose saturation is unreachable, or where a saturated count is a
   visible absurdity rather than silent corruption — which is what a verifier
   wants.

3. **A plain operator under an `#[allow]` carrying an `// ARITH:` comment that
   proves it cannot overflow.** The comment is not optional and it is not
   "this looks fine": it names the invariant.

   ```rust
   // ARITH: `shift` is compared against 28 before every increment, so it
   // never leaves 7..=35.
   #[allow(clippy::arithmetic_side_effects)]
   fn read_vint(&mut self) -> Result<i32> { ... }
   ```

   Use this where `checked_*` would cost something real (a hot decode loop) or
   would obscure the code. Attach it as tightly as the language allows — a
   statement or a small function, not a whole module.

An `#[allow]` without an `// ARITH:` justification is a review failure. An
`#[allow]` at module scope in a gated crate is a `TODO(arith-audit)` marker
and nothing else.

### Two shapes that defeat their own guard

Both were found four times over in c27, in unrelated modules, which is what
makes them worth naming rather than leaving to notice:

- **`if a + b > len { reject }` where `a` came off disk.** The guard forms the
  very sum it exists to guard. `blocktree`'s `if fp + 8 > slice.len()` was the
  only bound on a `.tmd` file pointer: a negative one arrives as `usize::MAX`
  through `as usize`, `fp + 8` wraps to 7, the check **passes**, and the slice
  index panics. Write it as `a.checked_add(b).filter(|&e| e <= len)`, or
  compare against `len - a` where the code has already established `a <= len`.
- **Java's `>>>` ported as `>>`.** The unsigned shift is load-bearing wherever
  the operand can go negative: `SegmentTermsEnumFrame`'s
  `(start + end) >>> 1` keeps a binary-search midpoint correct past
  `Integer.MAX_VALUE`; `readVInt() >>> 1` and `token >>> 2` turn a corrupt
  chunk header into a large positive count rather than a negative one; and
  `DocIdsWriter.readInts21`'s `(int) (l >>> 42)` is a zero-extended 22-bit
  field, where a signed shift yields a *negative doc id* — a different answer,
  not a rejected one. Grep the Java for `>>>` and check every occurrence
  against the Rust; a signed shift there is usually a silent wrong answer
  rather than a panic, which is the harder half of this defect class to find.

### Test code

Tests, benches, examples and fixture builders opt out at their own boundary,
because the gate is about values read off disk and a test's `i + 1` is not
one:

- inside a `#[cfg(test)] mod tests` block: `#![allow(clippy::arithmetic_side_effects)]`
  as the first line of the block, with the one-line reason;
- in a `tests/`, `benches/` or `examples/` file: the same as a file-level
  inner attribute under the module doc comment.

This is what keeps the gate meaningful. A gate that fires on every `i + 1` in
test code is one that gets `#[allow]`-ed everywhere and stops meaning
anything.

## Two hand-checked rules the lint cannot express

**Both are now mechanically checked too** -- `c41-gates-and-record` turned them
into `scripts/check-port-invariants.py`'s `fixed-bitset-bound` and
`sentinel-callers` rules, both in `scripts/gate.sh` and CI. They stay written
out here because a gate is only as good as the understanding behind it, and
because **neither rule can check the thing that matters most**: whether an
`// FBS:` proof is true, or whether the check at a sentinel call site is the
*right* check. See [`docs/mechanical-gates.md`](mechanical-gates.md) for what
each one does and does not catch. On their first run they found five live
defects.


### Bound an index against the collection it indexes

**Never index a `FixedBitSet` with an index bounded against anything other
than that bitset's own `len()`.**

`FixedBitSet::get`/`set`/`clear` do `words[index >> 6]`, and until c41 the
only bound on that was a bare `debug_assert!(index < self.num_bits)` -- Java's
own `assert`, which is off in production. An index out of range but still
inside `words` therefore read or wrote a **ghost bit** past `num_bits` in a
release build: a silently wrong live/dead answer. **They now check the bound
unconditionally and panic**, so the silent half of the class is gone; what is
left is an aborted query where the answer should have been an error, and
`clippy::arithmetic_side_effects` still sees none of it -- this is the indexing
row of the table above, which the lint does not cover.

Two consequences worth stating, because they are why the rule did not go away
with the bound:

- A **wrong-but-in-range** index is still a wrong answer, and no bound check
  can see it. That is what the rule is for.
- The panic is a containable failure (`lucene_ffi`'s `guard` catches it), but
  on a search path it is still a dead query. **The bound belongs where the id
  is decoded, not where it is used**, and
  [`FixedBitSet::get_doc`](../crates/lucene-util/src/fixed_bit_set.rs) is the
  one-line way to spell it for an id that came from somewhere else -- a
  postings walk, a BKD leaf, a vector store's ordinal range. c41 migrated 30
  hand-written copies of that bound onto it.

The rule is deliberately mechanical rather than "bound your indices", because
the defect always looks correct locally. c28 found it twice in one crate, in
both directions:

- `term_delete::resolve_term_doc_ids` filtered postings through
  `live_docs.get(doc_id as usize)` with no bound at all -- and `as usize`
  sign-extends, so a negative doc ID from a corrupt `.doc` became
  `usize::MAX`.
- `deletes::mark_deleted` *did* bound the doc ID, but against `max_doc` -- a
  **separate caller-supplied parameter** from the `&FixedBitSet` it then
  indexed. Every caller in the port passes a consistent pair, so the code was
  correct; it was one caller away from not being, and the function's own doc
  comment promised `DocOutOfRange` "rather than ... panicking".

Both are now bounded against the bitset's own length, with the length hoisted
out of the loop. Grep for `bits2words`, `FixedBitSet` and `.get(` when auditing
a module: if the bound and the bitset do not come from the same place, it is
this defect. `scripts/check-port-invariants.py --only=fixed-bitset-bound` does
that grep, and c41's Tier-2 review is the reason it also follows closure
parameters and `if let Some(bits) = ..` bindings: the first version missed
those, which is roughly half of the real index sites in `lucene-search`.

### Check an out-of-domain sentinel at every call site, and list them per site

A function that returns a sentinel *outside* the domain of its result --
`-1` for "no next set bit", `-1` for "not found", `0` for "no such block" --
has to be checked by **every** caller, and an audit that records "this
function's sentinel is handled" instead of "this call site handles it" will
miss one. c31 hit exactly that: `bit_table_next_bit_set` returns `-1` for "no
next present arc", and the batch bounded its *upper* end (`index >= num_arcs`
-> `-1`) while leaving the sentinel itself unchecked at
`read_next_real_arc` -- so `-1` flowed on as an `arcIdx` and `read_arc`
derived `firstLabel - 1`, an arc one label below the range the node declared,
and for `firstLabel == 0` exactly `END_LABEL`. The sibling call site in
`read_next_arc_label` had the check; the batch's own report claimed both did.

Three things make this class hard to see and are worth doing deliberately:

- **A byte-flip sweep cannot find it.** The sweep asserts "a typed error or a
  clean decode", and a plausible wrong label is a clean decode. c31's sweep ran
  40 136 flips over this exact code and passed.
- **The consequence depends on data the fixtures do not vary.** `firstLabel`
  is an ASCII letter in every FST fixture in the tree, so `firstLabel - 1` is
  an ordinary label; only `firstLabel == 0` turns it into `END_LABEL`. Write
  the regression test at the value that makes the consequence loudest.
- **The sentinel often produces a *valid* address.** `presence_index` going
  `-1 -> 0` addresses `posArcsStart`, and a `-1` `presence_index` addresses
  *above* the arc array -- both inside the body, so the reader's own range
  check accepts them. "The bad value is caught downstream" is a claim to
  verify, not to assume.

So: grep for every call of a sentinel-returning function, and record the check
**per call site** in the sweep report. Java usually marks these with an
`assert` (`assert nextIndex != -1`), which is off in production -- so a
`debug_assert!` port is not a port of the check, it is a port of the *absence*
of one.

## Lints considered and not adopted

Measured across the workspace at c19 (`cargo clippy --workspace --all-targets
-- -W <lint>`, unique sites):

| lint | sites | verdict |
|---|---|---|
| `clippy::arithmetic_side_effects` | 2 063 (1 859 outside test modules) | **adopted, scoped.** A workspace-wide deny would need ~2 000 `#[allow]`s, which is the failure mode itself. Per-crate adoption with a per-module burn-down is what makes it survivable. |
| `clippy::indexing_slicing` | 2 371 + 371 slicing | **not adopted.** Covers a real half of the class (b4, b6) but is even noisier, and in `check_index.rs` alone it would rewrite 77 sites that index this port's *own* `Vec`s. Worth revisiting per module during an audit. |
| `clippy::cast_sign_loss` | 1 036 | **not adopted as a standing deny; adopted as a one-shot audit step.** It would have caught the `i32 as usize` sign-extension shape in b1/c15 directly, but it fires on every deliberate widening in the codebase. c27 assessed it per module on six freshly audited modules, which is the scoped form c24 recommended: `points` 37 sites -> **1 live defect** (a leaf pointer still sign-extending into `seek`), `blocktree` 44 -> ~5 candidates, all already guarded, `stored_fields` 56 -> 0 live, `term_vectors` 52 -> 0 live, `doc_values` 7 -> 0 live. In every one of those the remaining sites are deliberate bit reinterpretations -- `as u32`/`as u64` is how this port spells Java's `>>>`, `as u8` its byte truncation -- so denying it would cost 30-50 `#[allow]`s per module whose proofs restate proofs the arithmetic gate already carries. **The recommendation is therefore: run it once during a module's burn-down and fix what it finds, then leave it off.** The exception is a module where the count is genuinely tiny and the shape is worth locking against: `for_util` has 3 sites, all same-width `i32 as u32`, and now carries a module-scope `#![deny(clippy::cast_sign_loss)]` so a future `i32 as usize` in the decode kernel cannot land silently. |
| `clippy::cast_possible_truncation` | 1 242 | **not adopted**, same reason. |
| `clippy::disallowed_methods` on `Vec::with_capacity`/`reserve`/`resize` | 270 + 23 call sites | **not adopted.** This is the only shape that *aborts*, so it is the one worth catching most — but `clippy.toml` is workspace-global with no per-crate scoping, and most of the 270 are sized by this port's own in-memory data. Left as a review item; the audited modules cap every disk-sized reservation by hand. |
| `overflow-checks = true` in `[profile.release]` | — | **not adopted.** It converts a silent release wrap into a panic, which is a different failure rather than a fix, and it is not a clippy gate. |

## Enforced by

Two checks, both in the existing pre-commit gate:

- **`cargo clippy --workspace --all-targets -- -D warnings`** enforces the deny
  itself. No new command, no new CI job.
- **`python3 scripts/check-arith-allows.py`** enforces the *rule for switching
  it off*, which is where the value actually is. It requires every
  `#[allow(clippy::arithmetic_side_effects)]` under `crates/*/src/` to be
  either a `// TODO(arith-audit)` burn-down marker or preceded by a comment
  block containing `ARITH:`, rejects a module-scope `#![allow]` outside a
  `#[cfg(test)]` block, and checks the burn-down counts in the table above
  against the markers actually in the tree. It caught a real unjustified
  `#[allow]` on its first run.

What neither can check is whether an `// ARITH:` proof is *true*. c19's Tier-2
review found three that were not — a proof naming an invariant that a
`debug_assert!` does not actually enforce, one whose stated numeric range did
not establish its conclusion, and one describing a corruption that the code
above it makes unreachable. All three had correct conclusions and wrong
reasoning, which is the failure mode to watch for: the next person to touch
that code trusts the comment. Review the proof, not just its presence.
