# c28 — the arithmetic gate, burned down across 8 `lucene-index` modules

Follow-up batch applying to `lucene-index` what c24 applied to `lucene-codecs`.
c19 turned `clippy::arithmetic_side_effects` on crate-wide for `lucene-index`
and left **11 modules** carrying a `#[allow(...)] // TODO(arith-audit)` marker
on their `mod` declaration in `lib.rs`. This batch audits the 8 that are not
owned by another in-flight batch (`check_index`/`checksum_verify` are c25's,
`merge`/`merge_policy`/`index_writer` are c26's), following
`docs/arithmetic-gate.md`: prove it, bound it and make the failure a typed
error, or say it is infallible and name the check.

Per the mid-batch scope correction from c25, the audit is **not** limited to
what the lint reports. `clippy::arithmetic_side_effects` covers arithmetic and
shifts only; every `&slice[a..b]` / `slice[i]`, every
`Vec::with_capacity(n)` / `vec![0; n]` and every `copy_from_slice` whose
operand came off disk was hand-checked in each module as well. **Three of the
six CORRECTNESS findings below came out of that hand-check, not out of the
lint** — including the two abort-class ones.

Java read from **`/home/tuong/work/lucene-10.5.0`**, the pinned tag.

## Burn-down

| | count |
|---|---|
| modules carrying the marker at c19 | 11 |
| **audited this batch** | **8** |
| **remaining marked** | **3** (`index_writer`, `merge`, `merge_policy` — c26's) |

`docs/arithmetic-gate.md`'s `lucene-index` row reads 3, matching the markers
actually in `lib.rs`; `python3 scripts/check-arith-allows.py` reports no
`lucene-index` problems.

Audited: `buffered_updates`, `deletes`, `index_file_deleter`,
`indexing_chain`, `segment_info`, `segment_infos`, `term_delete`,
`update_document`.

Lint sites resolved: **42** — 38 in lib code, 4 in test modules (which take the
gate's `#![allow]`-at-the-block boundary, per `docs/arithmetic-gate.md`'s
test-code rule). Of the 38 lib sites, **29** carry an `// ARITH:` proof and
**9** are gone: 5 converted to checked or clamped arithmetic (F5, F7, F8 and
the refcount pair in F10), and 4 by hoisting a value that was being recomputed
— `inflate_gens` computed `gen + 1` six times across three comparisons and now
computes it once.

| Rust file | Java counterpart (10.5.0) |
|---|---|
| `segment_infos.rs` | `index/SegmentInfos.java`, `index/SegmentCommitInfo.java` |
| `segment_info.rs` | `codecs/lucene99/Lucene99SegmentInfoFormat.java` |
| `index_file_deleter.rs` | `index/IndexFileDeleter.java`, `util/FileDeleter.java`, `index/IndexFileNames.java` |
| `deletes.rs` | `index/ReadersAndUpdates.writeLiveDocs`, `codecs/lucene90/Lucene90LiveDocsFormat.java` |
| `term_delete.rs` | `index/FrozenBufferedUpdates.applyTermDeletes` (single-segment resolve half) |
| `update_document.rs` | `index/IndexWriter.updateDocument` + `SegmentInfos.changed`/`write` |
| `indexing_chain.rs` | `index/IndexingChain.java` (`PerField.invert`) |
| `buffered_updates.rs` | `index/{DocumentsWriterDeleteQueue,BufferedUpdates,BufferedUpdatesStream}.java` |

Findings: **6 CORRECTNESS**, **3 MISSING**, **2 PERF**, **2 INTENTIONAL**.
Eight of the nine CORRECTNESS/MISSING are fixed; the ninth (F9) is recorded
with the layering reason that makes it unfixable at this level.

On tests, precisely: **seven of the eight fixes have a test that fails against
the unfixed code**, verified by reverting each fix and watching it fail. The
exception is F7, whose test exercises a helper (`advance_position`) that did
not exist before the fix, so it cannot be run against the unfixed code at all
— the reachable-input path needs a ~2 GB field value. That is stated here
rather than glossed. **One module came out genuinely clean**: `update_document`
needed proofs only, no runtime change. `term_delete`'s *arithmetic* was clean
too — its one defect (F6) came out of the indexing hand-check the lint cannot
do.

---

## Findings

### F1 `[CORRECTNESS]` `segment_infos.rs` — every generation off `segments_N` is one `+ 1` away from a panic

`SegmentCommitInfo`'s three generations (`delGen`, `fieldInfosGen`,
`docValuesGen`) are read as raw big-endian `i64`s with no bound, exactly as
Java reads them. Java then derives the next one with `delGen + 1` in the
`SegmentCommitInfo` constructor, and `advanceDelGen()`/`advanceDocValuesGen()`/
`advanceFieldInfosGen()` each step by one more. In Java a `long` wrap is
silent; in Rust the bare `+` is a **panic in a debug build**, and in a release
build it wraps `i64::MAX` to `i64::MIN` — a *negative* generation, which
`liv_file_name` then formats as `_0_-9223372036854775808.liv`, a name no
Lucene can read and one this port's own deleter would not recognise as
belonging to the segment.

Every generation-consuming site in the crate is downstream of this: all three
`advance_*` methods and their `advance_next_write_*` twins,
`derive_next_gen`, `deletes::apply_deletes`' `next_del_gen + 1`,
`field_updates`' whole write path (it reaches the arithmetic only through
these methods, which is why that module was already lint-clean), and
`update_document`'s `generation += 1` / `version += 1`.

Two more values feed the same shape: the commit's own `version` (a raw
big-endian `i64`, stepped once per commit) and `counter` (a vlong, the next
segment-name counter). And `generation` itself, which is not in the file at
all — it is the `N` parsed out of the `segments_N` **file name** in base 36,
so a directory entry named `segments_1y2p0ij32e8e7` hands `parse` a
well-formed `i64::MAX` from outside the file entirely.

Fixed by capping all six on the way in. `segment_infos::MAX_GENERATION` is
`i64::MAX / 2`, and `check_generation` rejects anything outside
`-1..=MAX_GENERATION` as `Error::InvalidGeneration`. `-1` is Lucene's "no such
file" sentinel and `0` means "no generation suffix"
(`IndexFileNames.fileNameFromGeneration`); anything below `-1` would name a
file with a `-` in the generation, which that method's `assert gen > 0` says
cannot exist.

Half the `i64` range as the cap is the point, and it is what makes the
remaining `+ 1`s provable rather than merely plausible: a generation advances
at most once per file this port writes, so climbing from `MAX_GENERATION`
(4.6e18) to `i64::MAX` would take 2^62 further index writes. Every `+ 1` that
survives now carries an `// ARITH:` proof naming that constant.

The margin is **not** uniform across the six values, and it is worth stating
separately. For the five *generations* it is enormous: each is bumped once per
delete or doc-values-update round, so a real index carries a value in the
tens. For `version` it is much tighter, because Java exposes
`SegmentInfos.advanceSegmentInfosVersion(long)` for an embedder to set it
directly, and the conventional thing to set it to is a nanosecond epoch —
around 1.7e18, i.e. **within about 2.7x of the cap**, not 2^60. Still safely
under it (and this port has no such API yet), but "unreachable by orders of
magnitude" is only true of the generations.

Deliberately **not** `saturating_add`: a saturated generation is the worst
possible outcome here, because two successive delete rounds would then write
to the *same* `.liv` file name — the second silently overwriting the first,
which is data loss rather than a crash.

**The cap alone was not enough, and the Tier-2 review caught why.** A cap
applied only on the way *in* leaves the derivation unclosed: every derivation
in this module is `+ 1`, so a commit read back at exactly `MAX_GENERATION`
derives `MAX_GENERATION + 1`, `advance_del_gen` stores it, `to_bytes`
serializes it — and the next `parse` of that commit **refuses it**. An index
this port wrote and can no longer open. The reachable route is F4's own threat
model: a directory entry named `_<base36(MAX_GENERATION)>.si` drives
`infos.counter` to `MAX_GENERATION + 1` through `inflate_gens`, and
`_0_<base36(MAX_GENERATION)>.liv` does the same to `nextWriteDelGen`.

Closed from both ends:

* `usable_generation` (the file-name path) is now **exclusive** of the cap —
  a name is untrusted input, so refusing the very top value costs nothing and
  stops the deleter manufacturing the boundary at all;
* `check_writable_generations` applies `check_generation` to all six values on
  the **write** path (`write_pending`), so a commit is refused rather than
  written unreadable. Refusing the commit is the honest failure: nothing is
  written, the previous `segments_N` stays current, and the caller is told
  which counter ran out instead of discovering it on the next open.

Pinned by `a_commit_this_port_writes_is_always_one_it_can_read_back`, the
property the whole cap has to satisfy — the boundary value round-trips, and
each of the six counters one past it is refused at write time with nothing
published. Both trash-name tests now use `base36(MAX_GENERATION)` rather than
`i64::MAX`, which is what made them miss this.

Tests: `absurd_generations_are_decode_errors_not_overflowing_next_gens` (all
three per-segment generations, plus the below-`-1` case),
`absurd_commit_version_and_counter_are_decode_errors`,
`absurd_file_name_generation_is_a_decode_error`,
`generation_at_the_cap_parses_and_still_derives_a_next_gen` (a cap off by one
would make `MAX_GENERATION` itself the panicking value), and
`a_commit_this_port_writes_is_always_one_it_can_read_back`.

One thing the proofs deliberately do **not** claim, since it is not true: the
generation fields are `pub` on a `pub` type and the three
`set_next_write_*_gen` setters take a bare `i64`, so nothing in the type
system re-establishes the bound between an `lucene-ffi` caller and
`advance_del_gen`. What is enforced is stated instead — the three *external*
paths (disk, file name, serialization) are gated, and
`debug_assert_generation` on the setters is the enforcer for the in-process
caller contract, with the write gate stopping a violated contract from ever
reaching a file.

### F2 `[CORRECTNESS]` `segment_infos.rs::parse` — `numSegments` and `numDVFields` sized reservations straight off the wire

`Vec::with_capacity(num_segments as usize)` on a value read as a bare
big-endian `i32`, and `Vec::with_capacity(num_dv_fields.max(0) as usize)` one
level down. `SegmentCommitInfo` is ~150 bytes of `String`s and `Vec`s, so
`i32::MAX` segments is a **~300 GB reservation**; the doc-values entry is a
`(i32, Vec<String>)` tuple, ~68 GB. Neither is a panic — an allocation failure
is an **abort**, which `catch_unwind` cannot intercept, so through the FFI it
takes the JVM down with no exception to catch. This is the same shape as
c24's `.fdm` 51 GB finding, in the very first file a `DirectoryReader` opens.

Only the *reservation* was unbounded: the loop that follows would have hit EOF
on the first iteration. Both counts are now checked against
`input.remaining()` before the reserve — a `SegmentCommitInfo` costs well over
one byte on the wire (a name, a 16-byte id, a codec name, three 8-byte
generations, two counts and two string sets), so a count above the bytes left
in the file is corrupt by construction and a well-formed file reserves exactly
what it did before. This is the same bound `lucene_store`'s own
`read_length` helper already applies to `readSetOfStrings`/`readMapOfStrings`.

`numDVFields < 0` is now an error too, rather than `.max(0)` quietly producing
an empty map from a stream that has gone out of step — the "a check that
degrades to a default is indistinguishable from one that passed" shape c25
flagged. Java sizes a `HashMap` from the same unbounded value and gets a
catchable `OutOfMemoryError`; here it is an abort, so the bound is not
optional.

Tests: `absurd_segment_count_errors_instead_of_reserving_for_it`,
`absurd_doc_values_field_count_errors_instead_of_reserving_for_it` (both
`i32::MAX` and `-1`).

### F3 `[CORRECTNESS]` `segment_info.rs::parse` — `numSortFields` sized a `Vec<IndexSortField>` unbounded

The same shape in the `.si`: `Vec::with_capacity(num_sort_fields as usize)` on
a vint bounded only by `>= 0`. `IndexSortField` is a multi-`String` struct, so
`i32::MAX` of them is a hundreds-of-gigabytes reservation and an abort. Java
allocates an equally unbounded `SortField[]` but gets a catchable
`OutOfMemoryError` for it.

Fixed with the same `input.remaining()` bound: a sort field is at minimum a
provider-name string plus a field name, a direction byte, a missing-value
marker and a type byte, so it cannot be cheaper than a byte apiece.

Test: `absurd_sort_field_count_errors_instead_of_reserving_for_it`.

### F4 `[CORRECTNESS]` `index_file_deleter.rs::inflate_gens` — a trash **file name** could panic index-open, and it decides deletions

`inflate_gens` exists to push every generation counter past whatever a crashed
session left in the directory, so the next write cannot land on an orphan. It
takes those generations from **file names**: `parse_generation` reads
`_<seg>_<gen>.<ext>` in base 36, and `max_segment_name` reads the segment name
the same way. Java catches only `NumberFormatException` and calls what is left
"trash file: we have to handle this since codec regex is only so good".

A name that *parses* but carries an absurd value is the same kind of trash and
Java takes it at face value, because `genLong + 1` merely wraps there.
`_0_1y2p0ij32e8e7.liv` is a perfectly well-formed base-36 `i64::MAX` in
exactly Lucene's own file-name shape, and any process that can create a file
in the index directory can produce one. Here that `+ 1` was a **panic in a
debug build**, at index-open time, before any query runs — and following the
value instead would have set `nextWriteDelGen` to a generation from which the
*next* `+ 1` cannot return.

Fixed with a `usable_generation` filter at the two places a generation is
parsed out of a name (`parse_generation`, and the segment-name/commit-name
scan), accepting only `0..=segment_infos::MAX_GENERATION` and discarding the
rest exactly as Java discards an unparsable one. The three `gen + 1`
comparisons per segment were also hoisted into one `let next = gen + 1` — they
each computed it twice before — and `max_segment_name.saturating_add(1)`
became a proved `+ 1` on a now-capped value, which is strictly better: the
saturation it replaced would have pinned `counter` at `i64::MAX` and made
every subsequent segment name collide.

Tests: `a_trash_file_name_claiming_an_absurd_generation_is_ignored_not_followed`
(both `i64::MAX` and `i64::MIN` in base 36, alongside a real `_0_2.liv` that
must still win), `trash_segment_and_commit_names_do_not_inflate_the_commit_counters`.

### F5 `[CORRECTNESS]` `deletes.rs::apply_deletes` — `delCount + newlyDeleted` overflowed before the bound that was meant to catch it

`sci.del_count` comes off `segments_N`, where the only bound this layer can
apply is `>= 0`: Java's `delCount > info.maxDoc()` check needs the `.si`,
which `segment_infos` deliberately does not read (see F9). So
`sci.del_count + newly_deleted as i32` on a `del_count` near `i32::MAX` is a
**panic in a debug build**, which is the whole reason the fix is needed. The
release build is less bad than it first looks and the report should say so: the
wrap produces a *negative* count, and `new_del_count as usize` sign-extends
that into a huge `usize` which the existing `> max_doc` test does catch — on
a 32-bit target too, where even `i32::MIN as usize` is 2147483648 and exceeds
any real `max_doc`. So in release the old code reported the right error for
the wrong reason. The fix makes the check the one that is actually doing the
work, and removes the debug panic.

Fixed by folding the widening and the addition into one checked expression
whose `None` reports `DelCountExceedsMaxDoc` with a saturated `i32::MAX` — a
visible absurdity against any real `maxDoc`, which is what the caller needs to
see. Test: `del_count_overflow_is_an_error_not_a_wrap`.

### F6 `[CORRECTNESS]` `term_delete.rs::resolve_term_doc_ids` — an out-of-range doc ID from a corrupt `.doc` was an index panic

Not a lint finding: this is the indexing half of the class, which
`clippy::arithmetic_side_effects` does not cover.

The live-docs filter was `live_docs.is_none_or(|bits| bits.get(doc_id as
usize))`. `FixedBitSet::get` indexes `words[index >> 6]`, which **panics in
release as well as debug** for a doc ID past the bitset — and `doc_id as
usize` sign-extends, so a negative doc ID from a corrupt postings list becomes
`usize::MAX` rather than anything a `< max_doc` test would catch.

What makes it a real inconsistency rather than a theoretical one: the very
same doc ID, with `live_docs == None`, reaches `deletes::mark_deleted`, which
reports it as `DocOutOfRange`. So whether a corrupt `.doc` produced a typed
error or a panic depended on whether the segment happened to have deletions
yet.

Fixed by resolving each doc ID through `usize::try_from` plus an explicit
`< bits.len()` bound and reporting the same `DocOutOfRange` the apply half
does, with the bound hoisted out of the loop rather than reloaded per doc. The
`live_docs == None` case takes an early return that hands back the decoder's
list and lets `mark_deleted` do the checking — the same single `Vec`
allocation the old filter chain made, without a per-doc predicate.

**The Tier-2 review found the same shape surviving in the apply half**, which
this batch had walked past: `deletes::mark_deleted` bounds `doc_id` against
`max_doc` and then indexes `bits` — two *separate* caller-supplied parameters.
Every caller in this port passes a consistent pair, so the code was correct;
it was one caller away from not being, and the function's own doc comment
promises `DocOutOfRange` "rather than ... panicking". Now bounded on
`bits.len()` (hoisted), the same way.

Two instances of one shape in one crate is what makes it worth a rule rather
than two fixes, so `docs/arithmetic-gate.md` now carries one: **never index a
`FixedBitSet` with an index bounded against anything other than that bitset's
own `len()`.** It is greppable, it is exactly the indexing row the lint does
not cover, and it would have caught both of these together.

Test: `a_doc_id_past_the_live_docs_bitset_is_an_error_not_an_index_panic`,
covering both failure modes `FixedBitSet::get` actually has — an empty bitset
(`bits2words(0) == 0`, so `words` is empty and *any* ID makes `words[id >> 6]`
an out-of-bounds index: the release panic) and a one-word bitset shorter than
the segment (the ID stays inside `words`, so release silently reads a **ghost
bit** past `num_bits`). The first version of this test used doc ID 2 against a
2-bit bitset, where `2 >> 6 == 0` — so it only ever tripped the
`debug_assert`, never the release-mode index panic its own comment claimed.


### F7 `[MISSING]` `indexing_chain.rs` — neither of Java's two position guards was ported

`IndexingChain.PerField.invert` does `invertState.position += posIncr` and
then guards it twice: it detects the `int` wrap after the fact (`if
(invertState.position < invertState.lastPosition)` → `IllegalArgumentException
("position overflowed Integer.MAX_VALUE")`) and separately rejects anything
past `IndexWriter.MAX_POSITION` (`Integer.MAX_VALUE - 128`, the headroom the
postings codecs need for their own sentinels). This port had the `+=` and
neither guard, and in Rust the bare `+=` is a **panic** in a debug build
rather than Java's wrap.

Reachability is narrow but real: `Analyzer::analyze` produces at most one
token per input byte and increments bounded by the token count (the stop
filter's accumulated gap is the largest), so it takes a single field value
carrying 2^31 token positions. That is a 2 GB string, not an impossibility,
and Java treats it as a document-level error rather than a process-level one.

`invert_documents_with_payloads` is infallible by signature (it returns an
`InMemoryInvertedIndex`, and its only caller is in `index_writer.rs`, which
c26 owns), so the guard is factored into `advance_position` and clamps at
`MAX_POSITION` rather than raising Java's exception. Clamping is the
conservative direction and not a silent wrong answer: positions only ever
collapse *together* at the ceiling, so the worst case is a false positive from
a phrase query on a document past 2^31 positions — where a wrapped negative
position would instead be encoded as a garbage vint delta and corrupt the
`.pos` file for every document after it. Raising the error properly needs
`invert_documents*` to return `Result`, which is a one-line change at the
single call site in c26's file and is recorded here rather than done across
a batch boundary.

Test: `position_accumulator_clamps_instead_of_overflowing` (including that the
ordinary `-1 → 0 → 1` seed and a synonym's zero increment are untouched).
This is the one test in the batch that **cannot** be run against the unfixed
code: it exercises `advance_position`, which the fix introduced, and reaching
the branch through `invert_documents` needs a ~2 GB field value.

Because the outcome diverges from Java — Java rejects the document, this port
indexes it with collapsed positions — the divergence is recorded in
`docs/parity.md` on the `IndexingChain` row, not only here. A false positive
from a phrase query on such a document is a silently wrong answer, and a
batch report is the wrong place for that to live alone.

One loose end worth naming: `check_index.rs` already carried a private
`MAX_POSITION` for the *read* side of the same rule
(`postings.positions_valid`), with a doc comment saying a position past it
"could only come from a corrupt `.pos` or a writer that never checked" — which
was an accurate description of this port until now. The two should be one
definition, and the writer's (`indexing_chain::MAX_POSITION`) is the one to
keep; `check_index` is c25's file, so the consolidation is recorded as a
one-line follow-up rather than done across the batch boundary.

### F8 `[MISSING]` `buffered_updates.rs::skip_sequence_numbers` — a backwards jump reissued a sequence number

`self.next_seq_no += jump` on a caller-supplied `i64`. Java's
`nextSeqNo.addAndGet(jump)` has the same shape, and both accept a negative
jump — which **rewinds** the counter and hands the same sequence number out
twice. A sequence number is only meaningful as an order (`FrozenBufferedUpdates
::applies_to` compares them to decide which segments a delete reaches), so two
operations sharing one is indistinguishable from a reordering. Overflow is the
same shape from the other end: Java wraps to a negative that sorts below every
number already issued, and the bare `+=` here panicked.

Fixed: the jump is clamped to `>= 0`, the addition saturates, and the result is
capped at a new `MAX_SEQ_NO` (`i64::MAX / 2`, the same half-the-range headroom
`MAX_GENERATION` reserves). The cap is what makes `next_sequence_number`'s own
`+ 1` proof airtight rather than "no caller reaches it": with it, no jump —
however absurd — can leave the counter without 2^62 of room. This is one of the
two places in the batch where `saturating_*` is the honest semantics rather
than a reflex, and the ceiling is what keeps it from becoming the
self-consistent-wrong-answer shape: a counter stuck at `MAX_SEQ_NO` never
reissues a number, because the `+ 1` above it still works.

Test: `a_backwards_or_overflowing_jump_never_reissues_a_sequence_number`.

### F9 `[MISSING]` `segment_infos.rs::parse` — Java's `maxDoc`-relative deletion checks, recorded not fixed

Java's `readCommit` runs three checks this port cannot: `delCount >
info.maxDoc()`, `softDelCount > info.maxDoc()`, `softDelCount + delCount >
info.maxDoc()`, plus a commit-wide `totalDocs > IndexWriter.getActualMaxDocs()`.
All four need each segment's `maxDoc`, which lives in its `.si` — and Java has
it because `readCommit` opens every `.si` inline, while this module
deliberately does not take a `Directory` dependency (its doc comment says so,
and `SegmentCommitInfo` deliberately does not own a parsed `SegmentInfo`).

Recorded, not fixed: the layering is intentional, the `>= 0` half of each
check is ported, the error message names the gap explicitly ("vs maxDoc
unknown at this layer"), and `check_index` (c25's file) verifies the
`maxDoc`-relative half where it does have both. F5 is what stops the missing
bound from becoming an overflow.

### F10 `[PERF]` `index_file_deleter.rs` — the refcount arithmetic, and where saturation is right

`ref_counts: HashMap<String, u32>`: `*count += 1` on incRef, `*count -= 1` on
decRef under a `debug_assert!(*count > 0)`. Both are now saturating, and the
reasoning differs per direction, which is why they are not a reflex:

- **incRef.** 2^32 live references to one file is unreachable (each is a
  commit point or a checkpoint holding the name in memory), and a saturated
  count only ever *keeps a file alive* — the safe direction. A wrap to 0 would
  delete a file the current commit still names, which is the data-loss
  outcome this module exists to prevent.
- **decRef.** Java's `RefCount.DecRef` asserts `count > 0` and, with assertions
  off, lets the count go negative — after which the file can never reach 0
  again and is leaked. A `u32` would wrap to `u32::MAX`: the same leak by a
  different route. Saturating leaves the count at 0, which is exactly the
  "known but unreferenced" state `open`'s init sweep already reclaims, so the
  file is deleted rather than leaked. Deleting a file at refcount 0 is not
  data loss by definition; only a caller bug reaches the branch at all, and
  the `debug_assert` is what catches that in tests.

No measurable cost: both compile to an add plus a `cmov`, on a path that runs
once per file per checkpoint.

### F11 `[PERF]` `indexing_chain.rs` — the per-token guard costs nothing measurable

F7 puts a `checked_add` plus one compare into the per-token loop, which is the
hot path the ~21 µs/doc figure comes from. A/B on `index-bench` (20 000 docs,
8 runs each, medians): bare `+=` **24 693 docs/s**, guarded **26 733 docs/s** —
i.e. the guard is well inside the run-to-run spread (23.6k–28.1k across the
16 runs) on a machine under heavy concurrent load. The two extra instructions
sit next to a `HashMap` probe and a `String` allocation per token, so this is
the expected result rather than a surprising one. No regression.

`ram_bytes_used`'s six `+=`/`*` sites were left as plain operators under one
function-level `// ARITH:` proof rather than made checked: each addend is the
size of a live allocation, so the total is within a small constant factor of
the bytes the process holds, and the function walks every term in the segment
once per flush.

### F12 `[INTENTIONAL]` `inflate_gens` tolerates a segment with no files on disk where Java asserts

Java's per-segment loop does `Long gen = maxPerSegmentGen.get(info.info.name);
assert gen != null; long genLong = gen;` — with assertions off, a segment named
by the commit whose files are all missing gives a `NullPointerException` on the
unboxing. This port uses `.unwrap_or(0)`, leaving that segment at its derived
generations. Kept: it is the same outcome Java's assertion is asserting *is*
impossible, and it is not the "degrades to a default instead of erroring"
shape — the value being defaulted is a *floor* to push counters past, so
defaulting it low can only fail safe (the segment keeps its own recorded
generation, which is already past everything the commit knows about). Already
covered by `inflate_gens_leaves_a_segment_with_no_files_on_disk_at_its_derived_gens`.

### F13 `[INTENTIONAL]` `mark_deleted`'s bitset is sized from `maxDoc`, matching Java

`vec![u64::MAX; bits2words(max_doc)]` is sized from the segment's `.si`
`docCount`, which `segment_info::parse` bounds only by `>= 0` — so up to
`i32::MAX`, a 268 MB allocation. Left alone: it is exactly
`new FixedBitSet(maxDoc)`, Java allocates the same 268 MB for the same input,
and `IndexWriter.MAX_DOCS` (2^31 − 129) caps what any writer can produce. This
is a large allocation, not an unbounded one — the abort class needs a value the
file chose *independently* of a real per-document cost, which this is not.

---

## Per-module verdicts

### `segment_infos.rs`
Java: `index/SegmentInfos.java`, `index/SegmentCommitInfo.java`.
**F1, F2, F9.** Two CORRECTNESS fixed (the generation cap, on all three of the
read, file-name and *write* paths; two unbounded reservations), one MISSING
recorded with the layering reason. Seven lint sites: six `+ 1`s on now-capped
generations under `// ARITH:` proofs anchored on `MAX_GENERATION`, one
in-memory `Vec` capacity. The write-side gate
(`check_writable_generations`) is new since the Tier-2 review and is what
makes the round trip closed rather than merely bounded. Swept.

### `segment_info.rs`
Java: `codecs/lucene99/Lucene99SegmentInfoFormat.java`.
**F3, F13.** One CORRECTNESS fixed (`numSortFields` reservation). Note the
write side of `segment_infos` still rejects only a *negative* generation, so a
caller could in principle hand `write` a generation `parse` would refuse —
Java has the same asymmetry (it validates neither on write), and nothing in
this port can reach it now that `inflate_gens` and `parse` both cap. The single
lint site is a `debug_assert`'s `buf.len() - FOOTER_LENGTH`, proved safe by
`check_footer` having already returned `Ok` — it `checked_sub`s the same
quantity and errors on the shorter file. Swept.

### `index_file_deleter.rs`
Java: `index/IndexFileDeleter.java`, `util/FileDeleter.java`,
`index/IndexFileNames.java`.
**F4, F10, F12.** One CORRECTNESS fixed (trash file names driving generations),
plus the refcount pair. Ten lint sites resolved. The file-name slicing
(`file_name[1..]`, `&rest[base_len..]`, `&file_name[..dot]`) was hand-checked
against arbitrary UTF-8 from a directory listing and is boundary-safe by
construction — the leading `_` is ASCII, `find`/`rfind` return boundaries, and
the byte-counting run only counts ASCII; the invariant is now written down
where `parse_segment_name` relies on it. Swept.

### `deletes.rs`
Java: `index/ReadersAndUpdates.writeLiveDocs`,
`codecs/lucene90/Lucene90LiveDocsFormat.java`.
**F5, F6 (second instance), F13.** One CORRECTNESS fixed (`delCount`
overflow), plus `mark_deleted`'s doc-ID bound moved onto the bitset's own
`len()`. Four lint sites: the
tail-word mask (`tail` is `max_doc & 63`, so `1..=63` inside the branch), the
`newly_deleted` counter (bounded by `max_doc`, since the bit is cleared in the
same breath), the generation `+ 1`, and the fixed one. Swept.

### `term_delete.rs`
Java: `index/FrozenBufferedUpdates.applyTermDeletes` (single-segment resolve).
**F6.** Lib arithmetic clean — both lint sites were in the test module's
fixture builder, which now takes the gate's block-level `#![allow]`. The one
defect here came from the hand-check the lint cannot do. Swept.

### `update_document.rs`
Java: `index/IndexWriter.updateDocument` + `SegmentInfos.changed`/`write`.
Lib arithmetic clean once F1's cap is in place: `generation += 1` and
`version += 1` are the two steps that cap exists for, and the `Vec` capacity is
an in-memory length. Test-module `#![allow]` added. Swept.

### `indexing_chain.rs`
Java: `index/IndexingChain.java`.
**F7, F11.** One MISSING fixed (both position guards), measured for
throughput. `ram_bytes_used`'s six sites carry one function-level proof.
Swept.

### `buffered_updates.rs`
Java: `index/{DocumentsWriterDeleteQueue,BufferedUpdates,BufferedUpdatesStream}.java`.
**F8.** One MISSING fixed (`skip_sequence_numbers`). The other five sites are
session-local monotone counters — `num_field_updates` (one per `Vec` push),
`BufferedUpdatesStream::next_gen` (one per flushed packet), and the sequence
number (one per indexing operation) — none read off disk, all under
`// ARITH:` proofs naming the step. Swept.

---

## Gates

- `scripts/verify-write-path.sh` — **ok (22/22 passed)**.
- `python3 scripts/check-parity.py` — ok, exit 0.
- `python3 scripts/check-arith-allows.py` — no `lucene-index` problems; the
  burn-down table's `lucene-index` row (3) matches the markers in `lib.rs`.
- `cargo test -p lucene-index` — 657 lib + 26 integration tests, all passing.
- `cargo fmt --all` — clean.
- Per-file line coverage, `cargo llvm-cov -p lucene-index`: `buffered_updates`
  98.01, `deletes` 98.60, `index_file_deleter` 99.07, `indexing_chain` 97.20,
  `segment_info` 98.42, `segment_infos` 97.95, `term_delete` 98.52,
  `update_document` 99.20 — every audited file above the 95% bar.
- `cargo clippy -p lucene-index --all-targets -- -D warnings` — clean for
  `lucene-index`; see the note below.

The workspace-wide `cargo clippy --workspace --all-targets -- -D warnings` is
currently red on two `clippy::clone_on_copy` warnings in
`lucene-search/src/doc_value_query.rs`, another batch's in-flight file;
`lucene-index` contributes zero diagnostics to it.

**Clippy note.** For most of this batch `cargo clippy` could not reach
`lucene-index` at all: `lucene-codecs` was mid-audit under c27 and failing its
own `arithmetic_side_effects` deny, and a dependency that does not compile
under clippy blocks every crate above it. The `lucene-index` lint state was
verified by re-running with `RUSTFLAGS="--force-warn=clippy::arithmetic_side_effects"`,
which downgrades the deny workspace-wide so every crate is linted and the
complete site list for `lucene-index` is visible: the 35 lib sites enumerated
above and nothing else, all annotated, plus the `examples/` and `tests/` files
that already carry c19's file-level opt-out.
