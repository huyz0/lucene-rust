# c34-ledger-reconcile — making the ledger true again

Follow-up batch on the record, not on a port. `docs/sweep/m2/LEDGER.md` carried
**68 items with an unticked `- [ ]` box**. Forty batches ran over many sessions,
and later batches repeatedly closed what earlier ones had recorded without
ticking the box — so the list the remaining work would be planned from was
claiming things a check disproves. That is the same defect class this sweep kept
finding in the code, and it is worse in the ledger, because a wrong record
sends the *next* batch to the wrong place. c26 and c29 each lost time to one.

**No Java counterpart exists for anything in this batch.** `docs/sweep/` is this
port's own record-keeping. Per the protocol's rule 1, no Java path is claimed.

Files touched: `docs/sweep/m2/LEDGER.md`, `docs/parity.md`,
`crates/lucene-index/src/index_writer.rs`,
`crates/lucene-codecs/src/{compound_format.rs,postings.rs,postings_writer.rs}`.

---

## 1. Counts

| | |
|---|---|
| Boxes examined | **68** |
| **CLOSED but never ticked** | **29** |
| **OBSOLETE** | **4** |
| **Genuinely OPEN** | **35 boxes / 33 distinct findings** (two findings have two boxes each) |
| Of those, closed by c34 as trivially closable | **2** |
| **Distinct open findings remaining** | **31** (in 32 boxes) |

Every one was verified against the tree — reading the code, grepping the
symbol, or running the test — never against the batch report that raised it.
That rule earned its keep: c31's report claimed a `debug_assert` conversion it
never made, and c30 found an earlier attempt's report full of `PLACEHOLDER`s
while its code was in fact landed. Two of the 29 closures below could only be
established by *running* something, and one open item could only be sized by
re-reading code whose own module doc was wrong.

## 2. The two things that could only be settled by running them

- **`corrupt_kdd_leaf_data_surfaces_as_points_error` fails on the current
  tree** (raised by b11). It does not.
  `cargo test -p lucene-search --lib corrupt_kdd` → `ok`. `points.rs` grew
  ~879 lines in b7/b8 and the entry assumed the corrupt `.kdd` the test builds
  had started decoding cleanly; the current `intersect` path rejects it.

- **`scripts/gen-fixtures.sh --check` was never run end to end** (raised by
  b10, deferred four times because the tree was never quiet). Ran it: **exit 0**.

  ```
  deterministic files verified byte-identical : 47
  non-deterministic files (random segment id) : 629
  deterministic mismatches                    : 0
  missing from committed tree                 : 0
  unexplained extras in committed tree        : 0
  manifests with a wrong key set              : 0
  segment-id baseline lines that disagree     : 0
  ```

  This also retires the item's premise. c32 rewrote `--check` to compare each
  manifest's *key set* (`blocktree_index/manifest.properties` is
  non-deterministic, so the old byte check was blind to exactly the damage c29
  hit) and to re-derive `fixtures/segment-ids.txt`. Both new checks pass.

And one that could only be settled by measuring: **coverage**. Two items —
"re-take `cargo llvm-cov --workspace`" (c1, which could only get a trustworthy
reading by pointing `CARGO_TARGET_DIR` at a scratch directory) and
"`lucene-search/src/lib.rs` sits at 90.5%" (c6) — were both live. Re-taken in
the container: **97.55% regions / 98.10% lines** workspace-wide,
`lucene-search/src/lib.rs` at **96.78% lines**, and **no file below 95% lines**.
The first item is now structurally closed as well: `scripts/gate.sh` ends with
`cargo llvm-cov --workspace --fail-under-lines 95`, so the reading is re-taken
on every commit rather than by hand.

## 3. Actively misleading entries

Four claimed a blocker that no longer existed, or described as structural
something that had since become local. Each is annotated in place; the pattern
is the finding, not the individual entry.

### 3.1 "Norms are opt-in per field … **blocked on** a multi-field `.nvd`/`.nvm` writer"

c26 wrote one. `norms::write_fields` (`norms.rs:392`) puts one or more norms
fields into a single `.nvm`/`.nvd` pair, `merge.rs:2022` calls it,
`Error::TooManyNormsFields` is gone, and `IndexWriter::add_norms_field`
accumulates. What is actually left is the opt-in itself:
`index_writer.rs:4789-4793` still forces `omit_norms = true` onto every indexed
field the caller did not name. **Nothing blocks closing it**, and it had been
sitting behind a false blocker for several batches. It is now the largest
ready-to-take item on the list.

### 3.2 `Lucene104PostingsWriter` impacts — and a module doc that contradicted the code

The ledger entry ("impacts are computed against norm 1") is *right*. What is
misleading is what a reader finds when they open the file: `postings_writer.rs`'s
module doc asserted, in two places, that

> impacts are always an empty byte region (no competitive-impact computation)

and

> since positions never co-occur with a full block in the first place
> (`total_term_freq < BLOCK_SIZE` is required whenever positions are indexed …)
> the level-1 entry's `indexHasPos`-gated pos/pay sub-fields are never
> reachable from this writer and are simply never written.

Both were made untrue by c20/c23 and neither was updated. The code writes one
`(maxFreq, norm = 1)` impact per level-0 block (`write_full_block`) and a
span-wide maximum per level-1 entry (`write_level1_span`, `postings_writer.rs:1283`)
— it has to, because real Lucene rejects a segment with *"Got empty list of
impacts"* — and `PosSkipWriter::write_level1` does write the pos/pay sub-fields,
which is what lets a positions-indexing field exceed `BLOCK_SIZE` at all.
**Corrected.** Anyone who had planned the impacts work off that doc would have
started by writing code that already exists.

### 3.3 "Stored-fields writer API takes `&[Document]` rather than streaming"

c4 made `StoredFieldsWriter` a real streaming object — `add_document(&Document)`
(`stored_fields.rs:1373`), `finish()` (`:1601`) — with `write_best_speed`
surviving as a convenience wrapper. Only the *read* half of the entry stands:
`document()` still materializes a whole `Document` where Java's
`StoredFieldVisitor` lets a caller take one field and skip the rest.

### 3.4 `docs/parity.md`'s FFI row — the last carrier of b12's F-7

The entry said the parity row had been corrected. Half of it had. The row for
`search/IndexSearcher.search(Query, Collector)` (single-segment, C-ABI) still
described `open_field_norms` as

> name → field number → `NormsEntry` → `FieldNorms::open`

long after b15 moved it onto `FieldNorms::from_field_stats`. `FieldNorms::open`
averages *decoded* (lossy above length 24) norms over *live* docs;
`from_field_stats` is Java's `sumTotalTermFreq / docCount` off the field's
`.tmd`, and `docCount` counts deleted docs. `avgdl` is in every score's
denominator, so this is a scoring claim, and the code had been right for
batches while the record was the only thing still asserting the divergence.
**Row corrected.** c34 also confirmed the code end to end: the nine surviving
`FieldNorms::open` call sites are all inside `#[cfg(test)]` modules, checked
against each file's own `#[cfg(test)]` line; the FFI and the benchmark runner
(`benchmarks/rust-runner/src/main.rs:478`) both use `from_field_stats`.

Two more were stale rather than misleading: the `DirectoryReader::open` entry
carried c1-era numbers (2.0 ms of 2.2 ms in `open_segments`) three batches out
of date — c12's `open_shared`/`SharedBytes` took it 579 µs → 120.7 µs on the
fixture corpus, and `88ebd47` landed after that, with `verdict-m1.6.md`'s
whole-corpus figure standing at 52.7 ms / ~155× — and the `.si`-rewrite entry
undercounted its own cost: **five** file groups now do the read-parse-rewrite-
fsync cycle, not four (`write_vector_files` joined them).

## 4. Two open items closed rather than handed back

Both were on the list only because the tree was never quiet enough; neither is
worth a batch of its own.

### 4.1 `lucene-codecs` privately duplicates primitives that exist in `lucene-store` (b1)

Most of it had already migrated (`block_packed.rs` uses
`lucene_util::zigzag`; `postings.rs` reads through `DataInput::read_group_vints`).
Two copies were left:

- `compound_format.rs` had its own `index_header_length` + `vint_len` pair,
  computed as `4 + vint_len(name) + name + 4 + ID_LENGTH + 1` against
  `codec_util::index_header_length`'s `9 + name + ID_LENGTH + 1 + suffix`.
  They agree for every codec name `write_index_header` accepts — ASCII shorter
  than 128 bytes, so the length prefix is exactly one byte — which is what made
  the swap safe. Both helpers deleted; the three call sites now use the shared
  one. The `vint_len` unit test is replaced by
  `index_header_length_matches_the_bytes_write_index_header_emits`, which pins
  the shared helper against the bytes `write_index_header` actually emits
  (including a 127-byte name, the boundary case) rather than against a second
  hand-derived formula. That is a stronger test than the one it replaces: the
  old one checked an arithmetic helper against itself.

- `postings.rs::write_group_vints` was a line-for-line copy of
  `DataOutput::write_group_vints` (`hnsw_vectors.rs` already used the trait
  method). Deleted; its 19 call sites moved onto the trait method.

### 4.2 Term-vector callers must supply fields in ascending field-*name* order (b7)

Raised by b7 with the fix assigned to "b9's flush path, where names are known".
It never landed, and the flush path did not do it: `build_term_vectors_output`
appended each field's vectors in `configs` order — described in its own comment
as "a stable, caller-controlled field order" — and `add_term_vector_field`
appends in call order.

Real `CheckIndex.checkTermVectors` walks `TVFields.iterator()`, which yields the
order the fields were written, and `checkFields` throws unless that order is
sorted **by field name**. The wire format carries field *numbers*, and this
writer's numbers come from the caller's field list, so number order and name
order need not agree. A caller who declared `zeta` as field 1 and `alpha` as
field 2, and configured them in that order, produced a segment real Lucene
rejects.

Fixed where the names are known: `build_term_vectors_output` sorts its
`TermVectorFieldConfig`s by name before building `per_doc`.
`write_best_speed`'s own caller contract is unchanged and still documented in
`docs/parity.md`; the row now also records that the `IndexWriter` flush path
satisfies it.

**The test is a tripwire, and was proved to be one.**
`term_vector_fields_are_written_in_ascending_field_name_order` builds exactly
that segment and asserts the written field numbers come back `[2, 1]`. With the
sort removed it fails with `left: [1, 2]` — checked, then restored. Its negative
control is the numbers themselves: they must stay 1 and 2, so this is a
reordering of the per-document field list, not a renumbering of the schema.

## 5. The four obsolete items

| Item | Why it no longer makes sense |
|---|---|
| `RegExp`'s `CASE_INSENSITIVE`/`CASE_INSENSITIVE_RANGE` and `DEFAULT_DETERMINIZE_WORK_LIMIT` (b8) | The determinize half is superseded — `regexp.rs`'s own gap list names exactly two gaps and that is not one of them; this port bounds *matching* instead, a different mechanism with the same visible contract. The case-insensitivity half stands but is unreachable: `RegexpQuery(Term)` passes match flags `0` and this port exposes no constructor taking them. An intentional divergence, recorded in the module doc, not outstanding work. |
| `postings_writer` should hold one `ForUtil` across blocks (b2) | There is no per-block `ForUtil` to hoist. b2 turned the encoder into free functions that pack in place with caller-supplied scratch; `postings_writer.rs` calls `for_util::for_encode`/`pfor_encode` directly (`:1189`, `:1225`) and never constructs a `ForUtil`. The struct still exists for the decode side; the writer does not touch it. |
| `c13` — c1's caller migration | A batch-shaped duplicate of two items that are each tracked separately and each still open (`try_*` migration; term-iteration/stats split). Its third part, `directory_reader → open_shared`, was done by c12. Keeping it is how the same work gets planned twice. |
| `docs/parity.md` has no mechanical staleness check (c12) | Superseded by a design decision that is written down. `scripts/check-parity.py` exists and *deliberately declines* to automate contradiction detection: its docstring records that a heuristic over the status text "flags fourteen of those for every real problem it finds", because a class routinely has several honest rows (read side and write side, a scoped first cut and a later widening). What it does instead has no false positives, and `--verbose` lists multi-row classes for a human. c12's three stale rows were fixed by hand; c34 fixed a fourth the same way. |

## 6. What changed in `LEDGER.md`

- A new **"Open work, prioritised"** section sits directly above the historical
  record and is now the list to plan from: 31 distinct open findings, grouped
  as **(A) wrong-answer bugs**, **(B) missing Lucene behaviour a caller can
  reach**, **(C) performance/memory divergence**, **(D) tooling and hygiene**,
  each stating what is missing, what it costs, and what blocks it.
- Everything below it is marked as the historical record, with a checkbox
  legend: `- [ ]` open, `- [x]` done, `- [~]` **obsolete**. Obsolete items are
  marked and explained rather than deleted, so the reasoning survives.
- All 29 closures carry a one-line note naming the batch or commit that did the
  work and the evidence checked — a symbol that now exists, a test that now
  passes, a number that was re-measured.
- Nine still-open entries that described a world one to three batches out of
  date carry a `**c34 restated**` note with what the tree actually looks like.

## 7. One observation worth carrying forward

Of the 29 closures, **eleven were closed by a batch that fixed the item as a
side effect of its own scope** — c6 closing four of b12's and b13's items while
wiring `FieldNorms`, c13 closing three FFI items while changing the boolean
ABI, c26 closing three merge items while building the format-coverage gate. In
none of those cases was the box ticked. The ledger's failure mode is not
carelessness about one's own work; it is that closing someone else's item is
invisible from inside the batch that does it.

The cheap countermeasure is the one this batch used and the one c26 built for
merge formats: when a batch removes a *blocker* rather than an item — a
one-field cap, a missing writer, an ABI shape — that is the moment to grep the
ledger for the word, because the entries that named it as their blocker are now
schedulable and nothing else will notice.

## 8. Gate

`python3 scripts/check-parity.py`, `python3 scripts/check-arith-allows.py` and
`python3 scripts/check-java-refs.py` all exit 0. `scripts/docker-test.sh gate`
green (fmt, clippy `-D warnings` on both targets, both arithmetic/parity/java-ref
checks, and `cargo llvm-cov --workspace --fail-under-lines 95`). Not committed,
per the batch instruction.
