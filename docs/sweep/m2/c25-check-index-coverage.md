# c25 — check-index coverage: firing the arms that had never fired, and deleting the ones that cannot

Follow-up batch. c19 took `check_index.rs` from 89.19% to **94.19%** — 0.81
short of this repo's 95% bar, its one unmet requirement — and named the reason:
roughly 150 `Check::fail`/`problems.push` arms in a 4 000-line verifier that
had **never once been observed to fire**. c19 closed the largest families with
twelve tests and enumerated the residual ~110 one-arm-per-test cases. This
batch finishes them.

Java read from **`/home/tuong/work/lucene-10.5.0`**, the pinned tag.

| Rust file | Java counterpart (10.5.0) |
|---|---|
| `crates/lucene-index/src/check_index.rs` | `lucene/core/src/java/org/apache/lucene/index/CheckIndex.java` (`checkFields`, `testPostings`, `testStoredFields`, `testTermVectors`, `testDocValues`, `testPoints`, `testSort`, `checkSoftDeletes`, `checkDocValueSkipper`) + `SegmentInfos.readCommit` |
| `crates/lucene-index/src/checksum_verify.rs` | `CheckIndex`'s checksum-only path + `CodecUtil.checksumEntireFile` (already over the bar at 97.11%; untouched) |
| `crates/lucene-codecs/src/term_vectors.rs` (F1, F2 — cross-batch) | `lucene/core/src/java/org/apache/lucene/codecs/lucene90/compressing/Lucene90CompressingTermVectorsReader.java` |

Totals: **3 CORRECTNESS** (all fixed; two found by this batch's own new
negative control, in a file this batch does not own; the third handed over by
c23), **10 arms deleted as unreachable by construction**, **25 new tests**, and
the coverage bar met.

The third CORRECTNESS finding is the one that matters most and it arrived
late, from c23: **a check that could not run was indistinguishable from a check
that passed.** It has its own section below.

---

## The rule this batch applies

c9 finding 9 established it and c25 extends it: **a shipped check must be
falsifiable.** An arm that cannot fire is not a safety net, it is a claim of
coverage that no input can redeem — and it is worse than no check at all,
because a reader auditing the verifier will tick it off as done. The two
shapes that make an arm unfirable here are:

1. **`x == x`** — the check compares two values the decoder derived from the
   *same* source. Java can perform the comparison because its `Terms`/
   `PostingsEnum` come from a codec that could in principle disagree with
   `FieldInfo`; this port's `blocktree`/`postings` decoders are *parameterized
   by* the `FieldInfos` the check would compare against.
2. **The decoder already rejected it.** `blocktree::open` and
   `hnsw_vectors::read_field_entry` validate on the way *in* where Java
   validates on the way *out*, so a file that would trip the check never
   reaches it — it is reported as `postings.open` / `hnsw.open` instead.

Ten arms fell to one of those two. They are deleted, with the reason left at
the site so the next reader does not re-add them.

---

## F0 `[CORRECTNESS → fixed]` a check that could not run read as a check that passed

**Handed over by c23**, which hit it from the other side: `field_infos::write`
put `omitNorms`/`storePayloads`/`storeTermVectors` on the wire for a
*non-indexed* field, where Java's `FieldInfo` constructor coerces them away.
Lucene coerces on read, so every cross-engine verifier stayed green — but this
port's own `field_infos::parse` rejects it, so `fnm.open` failed, and **every
postings check in the segment was then silently skipped**. c23 fixed the
`.fnm`. The verifier behaviour was mine.

### What was wrong

Almost every family in this module hangs off an `*.open` step, and when that
step failed the family contributed **nothing at all** to the `CheckResult` —
it simply was not in the list. Two consequences, the second much worse:

1. A caller counting failures saw *one* problem (`fnm.open`) where nine
   classes of invariant had gone unexamined. `all_passed()` was correctly
   `false`, but nothing said what had not been looked at.
2. Three term-vector families were pushed as **passes** in that state.
   `term_vectors.fields_marked_in_fnm`, `.self_consistent` and
   `.match_postings` are built from `problems.is_empty()`, and with no `.fnm`
   to name the fields the per-document loop skipped every field — so the
   problem lists were empty for the one reason that must never read as
   agreement: nothing had looked. `term_vectors.match_postings` is a
   *cross-check*, and a cross-check with one side missing reported a clean
   pass over a comparison nobody made.

And in two places `all_passed()` genuinely returned **`true`** over an
unverified segment. A `.si` that declares an index sort in a segment with no
doc-values files gets no `sort.docs_in_index_sort_order` at all, and
`fnm.doc_values_vs_files` does not cover it — a sort field absent from `.fnm`
leaves *no* field claiming doc values, so that check passes too. The declared
sort order that every merge, every early-terminating query and Lucene's own
`testSort` trust was then verified by nothing whatsoever, and the segment
reported as entirely healthy. Same shape for a `.fnm` naming a soft-deletes
field with no column behind it.

### The fix

`Check` now carries a three-valued [`Outcome`] — `Passed`, `Failed`,
**`Skipped`** — instead of a `passed: bool`. `Outcome::Skipped` means *this
check did not run because a named prerequisite failed*; it is not a pass, so
`all_passed()` is false and `failures()` includes it, and `CheckResult::skipped()`
lists exactly what went unguarded.

The distinction the model draws is narrow on purpose. A format the segment
legitimately does not have — no vectors, no points, a compound segment — still
produces **no check at all**, exactly as before, because there is nothing there
to be unguarded. Only a *failed prerequisite* produces a skip. That is what
keeps `every_real_lucene_index_fixture_passes_every_check` green and keeps the
new outcome meaningful.

### The audit: 17 prerequisites, 43 families

Every `*.open` arm was audited for this shape, plus the per-field
"the metadata parsed but has no entry for this field" arms, which are the same
thing one level down. All 17 now name their casualties, and the inventory is
pinned by `every_prerequisite_names_the_families_it_takes_down` so that adding
a new `*.open` arm without its `skip_families` call fails the suite.

| prerequisite | families it takes down | what that leaves unguarded |
|---|---|---|
| `si.open` | **16** (every family in the module) | the whole segment. The only prerequisite that can. |
| `fnm.open` | **12** — 9 by name plus the three term-vector families at their own site | c23's case. Postings, norms, doc values, points, vectors, HNSW, the index sort, the soft-deletes count, and the `.fnm`-vs-files cross-checks. |
| `postings.open` | **3**: `postings.*`, `term_vectors.match_postings`, `norms.agree_with_postings` | the two *cross-checks* are the sharp ones — each has a second, independent copy of the data on disk, and losing one side turned the comparison into a pass. |
| `liv.open` | **2** | `.liv`'s own size-vs-`maxDoc` and cardinality-vs-`delCount` checks. (The rest of the segment still runs, but believing nothing is deleted — recorded, not modelled as a skip, because nothing is skipped.) |
| `term_vectors.open` | **5** | the entire `.tvd`/`.tvx`/`.tvm` group. |
| `stored_fields.doc_count_matches_si` | **1** | `.fdt`'s per-document decode — and c25 measured **0** of 47 `.fdt` corruptions caught by any other check, so this is the file's only line of defence. |
| `doc_values.open` | **4** | the whole `.dvd`. c19 measured **0** of 261 `.dvd` corruptions caught elsewhere. |
| `doc_values.entry_present:<f>` | **3** (that field) | that field's whole doc-values column. |
| `norms.open` | **4** | the whole `.nvd`. c19 measured **0** of 99 `.nvm` corruptions caught elsewhere. |
| `norms.entry_present:<f>` | **2** (that field) | that field's norms — i.e. every BM25 score that reads them. |
| `points.open` | **5** | the whole BKD tree. |
| `points.field_present:<f>` / `points.decode:<f>` | **4** each (that field) | that field's leaves. |
| `vectors.open` | **6**, HNSW included | the flat vector store *and* the graph over it. |
| `vectors.field_entry_matches_fnm:<f>` | **2** (that field) | that field's vectors. |
| `hnsw.open` / `hnsw.open:<f>` | **3** each | the graph structure checks. |

**43 distinct family names**, of which 9 are the `<subsystem>.*` roll-ups the
`.si` and `.fnm` failures use — once the file that *names the fields* is gone,
the per-field expansion is unknowable, so the roll-up is the honest
granularity.

The three "0 caught by another check" rows are why this is worth the type
change rather than a comment: for `.fdt`, `.dvd` and `.nvd` a skipped family
is not a redundant one, it is the file's only reader.

### Tests

- **`a_prerequisite_failure_names_every_family_it_takes_down`** — c23's exact
  scenario, driven by an unparsable `.fnm`. Asserts every family below `.fnm`
  is present in the result and `Skipped`, that the three term-vector families
  that used to report **passes** are now skips, and — the other half — that
  `term_vectors.every_doc_decodes`, which needs no `.fnm`, is still a real
  pass. The skip is targeted, not a blanket "give up".
- **`a_declared_index_sort_with_no_doc_values_is_reported_as_unverified`** —
  the `all_passed() == true` hole, with the healthy control alongside. It also
  asserts that **nothing else fails**, which is the point: the skip is the only
  thing standing between that segment and a clean bill of health.
- **`every_prerequisite_names_the_families_it_takes_down`** — the inventory
  above, pinned.
- `corrupt_si_short_circuits_remaining_checks`, `missing_liv_file_fails_liv_open`,
  `a_compound_segment_skips_every_format_check` and
  `a_field_claiming_a_format_with_no_files_is_not_walked_twice` were all
  extended to assert the skips rather than the old silence.

---

## Deleted: ten arms that cannot fire

| # | arm | why it cannot fire |
|---|---|---|
| D1 | `postings.term_dict_shape`: `.tmd` `docCount > maxDoc` | `PostingsFileBytes::handles` passes `si.doc_count` to `blocktree::open` as `max_doc`, and `open` rejects a `docCount` outside `0..=max_doc` with `Error::InvalidDocCount`. The file is reported as `postings.open`. |
| D2 | `postings.term_stats`: `!hasFreqs ⇒ totalTermFreq == docFreq` | `blocktree`'s stats decoder sets `total_term_freq = doc_freq` verbatim when `index_options == Docs`. Both sides read the same `.fnm`. `x == x`. |
| D3 | `postings.doc_ids_valid`: `!hasFreqs ⇒ freq == 1` | `read_tail_block`/`refill_full_block` do `freqs.fill(1)` for a field without freqs, and the singleton path gets `freq = totalTermFreq = docFreq = 1` (D2). `x == x`. |
| D4 | `postings.field_summary`: `sumTotalTermFreq < sumDocFreq` | `blocktree::open` rejects exactly this with `Error::InvalidSumTotalTermFreq`, three lines after it reads the pair. |
| D5 | `postings.field_summary`: `!hasFreqs ⇒ sumTotalTermFreq == sumDocFreq` | `read_freq_pair` returns `(first, first)` for `IndexOptions::Docs` — the `.tmd` does not even carry a second value. `x == x`. |
| D6 | `norms.open`: `.nvm` format version vs `.nvd` format version | `norms::VERSION_START == VERSION_CURRENT == 0` and both headers are validated against that **one-element** range by `check_index_header`. Both versions are `0` or the open already failed. |
| D7 | `hnsw.neighbors_sorted`: neighbours *out of order* (`nbr < last`) | `OffHeapHnswGraph::neighbors_into` decodes a neighbour list as a running sum of **unsigned** deltas, so the list it returns is non-decreasing by construction. The *repeat* (`nbr == last`, a zero delta) still can happen and is kept. |
| D8 | `check_hnsw_graphs`' `node < 0 \|\| node >= size { continue }` | `read_field_entry` validates every upper-level node ordinal while parsing `.vem` (first non-negative, deltas non-negative, last below `size`), and level 0's node set is literally `0..size`. Not a reporting arm — a dead guard that only ever skipped nothing. |
| D9 | `postings.field_in_fnm`: a field with a term dictionary but **no** `.fnm` entry | `blocktree::open` resolves each `.tmd` record's field *number* through `FieldInfos::field_by_number` (unknown → `Error::InvalidFieldNumber`) and takes the field's **name** from what it found, so every name `iter_fields` yields came out of this very `FieldInfos`. `x == x`. The `indexOptions=None` half of the same check survives and is now driven. |
| D10 | `doc_values.skipper`: a fresh skipper must report `maxDocID(0) == -1` | Java is handed a `DocValuesSkipper` the codec produced, which a caller may already have advanced. Here it is constructed on the line above, and `DocValuesSkipper::new` sets every level's `max_doc_id` to `-1` unconditionally. |

D2, D3 and D5 together removed the **last** use of `has_freqs` in
`check_postings`, which is the finding underneath them: **the `.fnm`'s
frequency flag has no independent on-disk witness in this port at all.** Java's
`checkFields` cross-checks it three times; none of the three is portable here,
and `check_postings`' doc comment now says so at the site rather than in a
sentence a reader has to go looking for. (`hasOffsets`/`hasPayloads` *are*
checked, because those have a witness: whether the segment carries a `.pay`.)

D6 is a live tripwire rather than a permanent removal: a second norms format
version would make it a real check again, and the comment says so.

---

## Driven: the arms that *can* fire

Twenty-five new tests (twenty-one for the arms, four for F0's skip model).
The families, and what each one is:

### The commit header vs the segment (`SegmentInfos.readCommit`)

- **`a_commit_header_ahead_of_its_own_segment_is_caught`** — all three
  cross-`.si` header validations at once: a segment older than the commit's
  recorded `minSegmentLuceneVersion`, a segment older than
  `indexCreatedVersionMajor`, and a segment with no `minVersion` once
  `indexCreatedVersionMajor >= 7`. This port's `segment_infos::parse` never
  opens a `.si`, so these three are the **only** place the two headers are ever
  compared: a commit lying about what wrote it would otherwise parse, open and
  query perfectly. The test also pins the *agreeing* case, including that
  `indexCreatedVersionMajor < 7` correctly skips the last two checks.
- **`soft_del_count_larger_than_max_doc_is_flagged`** — `delCount`'s twin, plus
  the combined `delCount + softDelCount <= maxDoc` bound. `delCount` had a
  test; neither of these did.
- **`a_segment_name_that_is_not_base36_is_flagged`** — a name
  `updateMaxSegmentName` cannot parse. Every fixture in the repo is written by
  a real `IndexWriter`, so this had never been reachable from a fixture.
- **`a_commit_whose_segments_exceed_max_docs_is_flagged`** — LUCENE-6299's
  total-`maxDoc` bound, the one commit-level check that needs every `.si` to
  have been opened and is therefore appended by `check_directory` rather than
  computed in `check_commit`. The append itself was unverified.

### The per-format skip/report boundaries

- **`a_compound_segment_skips_every_format_check`** — this module has no
  compound-file support and every per-format check returns *silently* for a
  compound segment. That is a documented scope decision, and only one of the
  **seven** guards implementing it had ever executed. The other six could have
  been reporting spurious failures on every compound segment in existence.
- **`a_field_claiming_a_format_with_no_files_is_not_walked_twice`** — a field
  claiming doc values / norms / an index sort / soft deletes in a segment with
  no `.dvm`/`.dvd`/`.nvm`/`.nvd` is reported **once**, by `fnm.*_vs_files`, and
  the format's own walk then stays quiet. Vectors and points are the
  deliberate exceptions (neither has an `fnm.*_vs_files` counterpart, so
  `vectors.open`/`points.open` are the only places the absence can be
  reported) and the test asserts exactly that asymmetry, by name and in order.
  Both this test and the compound one **first prove the claims are live** —
  `check_field_flags_vs_files` must report all four families — because
  otherwise "nothing was reported" would also hold for a `FieldInfo` that
  claimed nothing, and the assertion would be vacuous. That is the c19 Tier-2
  lesson applied in advance.
- **`a_listed_but_missing_tim_is_reported_by_postings_open`**,
  **`vector_files_listed_but_absent_are_reported_by_their_own_open_checks`** —
  the `dir.open` failure arms, which are *different* arms from the ones c19's
  file-replacement control drives (those reach the readers' own parse
  failures). A file that is simply gone is the more common accident of the two.
- **`a_field_declaring_a_skip_index_with_no_dvs_file_is_caught`**,
  **`an_index_sort_naming_a_field_the_fnm_lacks_is_caught`**,
  **`a_soft_deletes_field_with_no_doc_values_entry_is_caught`** — three
  claim-without-data arms, one per subsystem.
- **`a_term_dictionary_for_a_field_the_fnm_says_is_not_indexed_is_caught`** —
  the surviving half of D9's pair. It is what a field-infos *update* generation
  that drops a field's `IndexOptions` looks like: every term in the dictionary
  becomes unreachable, because every reader asks whether the field is indexed
  before it opens the terms at all.

### The term dictionary's own claims

`postings_writer` computes every term statistic from the postings it is given,
so it **cannot be asked** for a dictionary that lies — which is precisely the
file `check_postings` exists to reject. Two new test-local helpers,
`patch_tim_stats` and `patch_tmd`, decode the written `.tim` term-statistics
region and the `.tmd` field-summary record, let a test change one claim, and
re-encode with the footer re-signed. That is c15's negative-control shape: real
writer output apart from the one semantic claim under test, so `file:*`'s CRC
cannot "catch" it and only the named check can fire.

- **`rewriting_the_tim_and_tmd_unchanged_leaves_the_segment_clean`** — the
  identity round trip, asserted byte-for-byte *and* through `check_segment`.
  Without it the four tests below would prove nothing: a test that fails
  because the re-encoding is broken says nothing about the check it names.
  (It also caught a real bug in the helper: `.tmd`'s trailing `indexLength`/
  `termsLength` are `writeLong`, which is **little-endian** in Lucene 9+ — a
  big-endian round trip is byte-identical and silently wrong.)
- **`every_per_term_statistic_bound_reports_its_own_violation`** —
  `totalTermFreq <= 0`, `totalTermFreq < docFreq`, and `docFreq <= 0`, each
  asserted by the specific message the arm emits.
  The `totalTermFreq < docFreq` case is the interesting one: `.tim` stores
  `totalTermFreq - docFreq` as an *unsigned* vlong, so the arm looks like
  another D2 — except this port's `read_vlong` accepts the ten-byte encoding
  whose top bit lands in bit 63 (a deliberate divergence from Java's cap,
  recorded in `data_input.rs`). So the check **is** falsifiable, and the test
  says in as many words that if `read_vlong` ever gains Java's cap, this arm
  joins D1–D10.
- **`per_term_total_term_freqs_that_overflow_i64_are_reported_not_summed`** —
  c19's F8 made this sum a `checked_add` precisely so a `.tmd` claiming
  `i64::MAX` could not be made to agree with itself. The arm that reports the
  overflow had never run, so F8's fix was itself unverified.
- **`a_tmd_max_term_below_the_dictionary_is_caught`** — `minTerm`/`maxTerm` are
  what `TermRangeQuery` pruning and `Terms.getMax()` read *without looking at
  the dictionary*, so a wrong one silently drops matches. Only the `minTerm`
  half had ever fired.
- **`a_positional_field_whose_segment_lists_no_pos_or_pay_is_caught`** — the
  two arms that *do* have an on-disk witness for the `.fnm`'s positional flags
  (`Lucene104PostingsWriter` writes a `.pay` exactly when some field indexes
  offsets or stores payloads, and a `.pos` exactly when some field indexes
  positions), plus the two decode paths that then decline to read positions
  rather than panicking.

### Per-document decode, driven by re-signed body corruption

- **`a_re_signed_body_corruption_is_reported_by_the_per_document_decode`** —
  Java's `testStoredFields` and `testTermVectors` both "decode every document,
  deleted ones included". Both walks existed here; both had only ever been
  observed *succeeding*. c19's file-replacement control reaches these
  subsystems by swapping a whole file, which fails at `open`; a flipped byte in
  the middle of a compressed chunk is the failure that gets *past* `open` — and
  past `file:*`, because the footer is re-signed — and lands in the per-document
  decode. **It found both of this batch's CORRECTNESS defects on its first
  run** (F1, F2 below).
- **`term_vectors_for_a_field_number_the_fnm_lacks_are_caught`** —
  `term_vectors.fields_marked_in_fnm` has two arms and c19's control drove only
  the first (`storeTermVectors=false`). The second is what a `.fnm` rewritten by
  a field-infos *update* generation looks like, and it is the worse of the two:
  the vectors are unaddressable rather than merely unexpected.

### Points and HNSW

- **`points_leaves_referencing_documents_past_max_doc_are_caught`** — Java's
  `VerifyPointsVisitor.visit` rejects both the per-point doc id and the
  field-level `docCount`; the two arms sit next to each other here and neither
  had fired, so a `.kdd` pointing past `maxDoc` (every `PointRangeQuery` on it
  then collecting garbage doc ids) was reported only by the weaker
  distinct-count mismatch.
- **`an_hnsw_entry_point_that_reaches_only_itself_is_caught`** —
  `hnsw.entry_point_reachable`'s one *failing* arm. Java reports connectedness
  without ever failing on it, and so does this port, except for the degenerate
  case: an entry point that reaches nothing but itself on a level with more
  than one node is not a quality issue, it is a graph whose search can never
  return more than one document however large `k` is. That distinction is the
  whole point of the check and the `.vex` corruption sweep cannot reach it —
  emptying the *entry node's* neighbour list specifically is not something a
  byte flip does. Driven through this port's own `.vem`/`.vex` writer with a
  hand-built `OnHeapHnswGraph`, with a connected positive control that differs
  in exactly one edge set, and an isolation assertion (an isolated entry point
  must not be reported as an out-of-level or repeated neighbour).

---

## Findings

### F1 `[CORRECTNESS → fixed]` a `.tvd` `prefixLength` sliced the previous term unbounded

**Where**: `crates/lucene-codecs/src/term_vectors.rs`, `build_field`.

`term.extend_from_slice(&previous_term[..prefix_len])`, where `prefix_len` is a
vint read off `.tvd` and `previous_term` is the *previously decoded* term — a
completely different part of the file, with nothing on the wire relating the
two. A corrupt `.tvd` names a prefix longer than the term it shares with and
the slice **panics**: `range end index 3 out of range for slice of length 0`.

A verifier that panics on a corrupt file has failed at its one job, and the
same decoder is on the query path (`term_vectors_query`, the highlighter), so
through the FFI this is a dead JVM.

**Fixed**: `previous_term.get(..prefix_len)` with a reported
`Error::Corrupted` naming both lengths. Test: the `.tvd` half of
`a_re_signed_body_corruption_is_reported_by_the_per_document_decode`, which
aborts the test binary without the fix.

### F2 `[CORRECTNESS → fixed]` a `.tvd` chunk's claimed decompressed length sized a `vec![0u8; n]`

**Where**: the same file, `read_chunk`.

```rust
let decompressed_len = (total_suffix_len + total_payload_len) as usize;
let mut decompressed = vec![0u8; decompressed_len];
```

Both halves are sums of per-term lengths read off `.tvd`, with nothing on the
wire relating them to the compressed bytes that follow. A single re-signed byte
flip produced `memory allocation of 1011335590898973 bytes failed` — a
**SIGABRT**, which is the one failure shape `catch_unwind` at the FFI boundary
cannot intercept (`docs/arithmetic-gate.md`'s fourth row). This is c19's F2
class, one file further out.

**Fixed**: the LZ4 block format expands by at most 255x (one `0xFF`
length-extension byte yields 255 output bytes), so a decompressed length past
that multiple of the compressed bytes left in the chunk cannot be produced by
*any* input and is rejected before a byte is allocated. The sum itself is a
`checked_add` + `usize::try_from`. `build_field`'s indexing of the flat
position/offset/payload streams was hardened in the same pass — each per-term
`freq` now slices its stream up front, which is what bounds the three
`Vec::with_capacity(freq)` calls next to it, and the per-term array reads are
`get(..).ok_or(Eof)` rather than `[idx]`.

Both are cross-batch edits to `lucene-codecs`, which **c24-arith-codecs**
owns; they are recorded under Handoffs. `term_vectors.rs` no longer carries a
`TODO(arith-audit)` marker, i.e. c24 has already audited it under the
arithmetic gate — which found neither of these, because the gate covers
arithmetic and shifts and *not* indexing or allocation sizing (the two rows
`docs/arithmetic-gate.md` explicitly records as uncovered). That is the
concrete case for the two lints it lists as "considered and not adopted".

---

## Per-file rejection rates (extending c19's table)

Every row is a **re-signed** sweep: the footer is recomputed over each
corruption so `file:*`'s CRC cannot claim the catch and only semantic checks
can fire.

| control | corruptions | rejected by the named check | by another check | accepted | isolated case? |
|---|---|---|---|---|---|
| `.nvm` → `norms.*` (c19) | 99 | **85** | 0 | 14 | yes |
| `.tip` → `postings.seek_agrees` (c19) | 99 | **44** | 12 | 43 | yes |
| `.vex` → `hnsw.neighbors_*` (c19) | 318 | **138** | 3 | 177 | yes |
| `.dvd` sorted → `doc_values.*` (c19) | 99 | **18** | 27 | 54 | yes |
| `.dvd` numeric+binary → `doc_values.*` (c19) | 261 | **69** | 0 | 192 | yes |
| `.doc` → `postings.advance_agrees` (c19) | 2 034 | **5** | 2 026 | 3 | no |
| **`.fdt` → `stored_fields.every_doc_decodes`** (c25) | **47** | **33** | **0** | **14** | n/a (0 elsewhere) |
| **`.tvd` → `term_vectors.every_doc_decodes`** (c25) | **43** | **15** | **21** | **7** | n/a |

Two of the new numbers are findings in their own right:

- **`.fdt`: 0 caught by another check.** Nothing else in the module reads the
  stored-fields payload, so a corruption this walk misses is a corruption
  nothing catches — the same structural fact c19 recorded for `.nvm` and
  `.dvd`, and the test now asserts `caught_by_other == 0` so it stays true. The
  14 it misses are stored-field *values*: they have no second copy anywhere on
  disk, so a different value is still a valid one. That is the `.tip` phenomenon
  and it is correct, not a gap.
- **`.tvd`: 21 of 43 caught by *another* check.** The term-vector subsystem is
  the one place in the module where two independent copies of the same data
  exist on disk, so `term_vectors.self_consistent` and
  `term_vectors.match_postings` see most corruptions before the decode walk
  does. That is the cross-check earning its keep, and it is why the floor for
  this row is 12 rather than something near 43.

---

## Coverage

Measured with `cargo llvm-cov --no-fail-fast -p lucene-index --lib`, run into a **private `CARGO_TARGET_DIR`**
(`target-c25-cov`), after `cargo llvm-cov clean --workspace` on that directory.
That is not incidental: `cargo llvm-cov` merges every `*.profraw` under the
target dir, so a concurrent batch measuring in the same worktree silently
poisons the result — c19 recorded a run that reported `check_index.rs` at 61%
for this reason, and three other batches were running in this worktree
throughout c25.

| file | c19 | c25 | bar |
|---|---|---|---|
| `crates/lucene-index/src/check_index.rs` | 89.19% → 94.19% ❌ | **97.25%** (196 missed of 7 134; regions 96.88%) | ✅ |
| `crates/lucene-index/src/checksum_verify.rs` | 97.11% | **97.11%** (untouched) | ✅ |

The run: `643 passed; 0 failed`.

### Arm count

Counted as *arms* (one `Check::fail`/`problems.push` site, or one
skip/report guard), not as lines:

- **Driven for the first time: 45.** They are the 21 arm tests' subjects,
  listed by family above. (F0's four tests are about the *result type*, not
  about firing an arm, so they are counted separately.)
- **Deleted as unreachable by construction: 10** (D1–D10).
- **Remaining: ~39**, spread over **110 uncovered production region starts**
  (down from 246 — the same measure c19's "139 uncovered region starts" used).
  Enumerated below with the reason each is still open — none is "forgotten",
  and one of them is a candidate for the D-list once someone proves the
  reachability either way.

  The count went 246 → 88 → 110 because **F0 added arms of its own**: each
  `skip_families` call is a new never-fired arm until a test makes its
  prerequisite fail. Most are covered (the `si`/`fnm`/`liv`/`term_vectors`/
  `doc_values`/`norms`/`points`/`vectors` open failures all have tests); the
  residue is the per-field skip sites whose prerequisite this batch has no
  test for — `points.decode:<f>` and `hnsw.open:<f>` — which are the same two
  arms already on the remaining list below. Reporting the higher number rather
  than the pre-F0 one is the honest thing: the module got larger and better,
  not smaller.

A further **49 region starts are in the test module**: they are the
`assert!(cond, "…{:?}", x)` message closures, which by construction only
evaluate when a test fails. They are permanently unreachable denominator, and
they grow with every test a batch adds — the small tax a coverage batch pays
on its own number (c19 recorded the same effect, at 40 lines).

### What is left, and why

| family | lines | why it is still open |
|---|---|---|
| `check_one_vector_field`'s memo/decline paths (a vector field with positions in a segment with no `.pos`, a `try_seek_ceil` error, an `Ok(None)` decode, the memo-eviction refill) | 11 | Reachable; each needs its own hand-built term-vector *and* postings pair, which is the most expensive fixture shape in the file. Two of them (the memo refill) are covered by other modules' tests and only show as missing under this batch's filtered measurement. |
| `postings.seek_agrees`' three `seekCeil` disagreement arms, `postings.advance_agrees`' re-seek `Err`, `compare_intersect_with_scan`'s mismatch arm | 8 | Reachable in principle and **not** reachable by byte corruption: c19's `.tip` sweep of 99 re-signed corruptions produced none. They need a trie that resolves to a *different but still valid* term, which a flipped byte essentially never yields. Closing them means a hand-built `.tip` that disagrees with its `.tim` — the `patch_tmd` treatment applied to the trie, which is a much larger helper than the two this batch added. |
| `doc_values.values_decode`'s per-document `Err`/ordinal-range arms and `doc_values.terms_sorted`'s `valueCount` mismatch | 30 | The largest remaining block. Needs a `.dvm`/`.dvd` pair that *decodes* but disagrees — an ordinal outside the terms dictionary, a non-monotonic SORTED_SET ordinal run, a dictionary whose size disagrees with `valueCount`. That is a `patch_dvm` helper in the shape of this batch's `patch_tmd`, with one entry layout per doc-values type. The clearly-shaped next batch. |
| `vectors.values_decode`'s **byte-encoding** arms and `check_ord_to_doc`'s two error arms | 16 | No byte-encoded vector fixture exists anywhere in the repo; every `.vec` here is `Float32`. Either a `GenVectorsByte.java` fixture or a hand-built one through `vectors::write`. |
| `points.decode`, `points.leaf_bounds_subset_of_field`, `points.doc_count_matches`' `docCount > pointCount` | 13 | The leaf-bounds arm needs a field with **more than one index dimension**, which `write_points_fixture` does not build (it is hardcoded to `numDims = numIndexDims = 1`); the other two need a `.kdm`/`.kdd` that disagree. |
| `hnsw.open:<f>` (per-field graph error), `sorted_nodes_on_level` error, `check_hnsw_graphs`' no-graph-files return, `connected_nodes_on_level`'s out-of-range entry | 8 | The first two need a `.vem` that parses but whose per-field graph does not. `connected_nodes_on_level`'s `entry < 0 \|\| entry >= size` guard is a **D-list candidate**: `entry_node` is validated by `read_field_entry`, so it can only fire for a field with `size == 0` and a non-zero `vectorIndexLength`, and whether `read_field_entry` rejects that pair is not proven either way here. |
| singletons: `sort_key_values`' non-numeric doc-values type, `check_field_norms`' empty-field and no-term-dictionary paths, `open_postings_bytes`' single-underscore suffix branch, `check_doc_values`' `unreachable!("filtered above")` | 6 | One line each, each needing its own segment shape. The `unreachable!` is required for match exhaustiveness and can never execute. |
| `check_segment_vs_commit_header`'s closing brace and a handful of `named_field_check` region boundaries | 3 | Region-attribution artefacts rather than real arms — llvm-cov starts a region on a closing brace of an `if let Some(..)` whose `None` side is unreachable. |

---

## Runtime

**Of the verifier: unchanged.** Nothing was added to any per-document,
per-term, per-ordinal or per-byte loop; the ten deletions remove work rather
than adding it, F2's guard is one multiply and one comparison per `.tvd`
*chunk*, and F0's skips are pushed only on a path that has already failed. `every_real_lucene_index_fixture_passes_every_check` —
`check_directory` over every Java-written fixture in the repo — runs in
**0.23 s** in release, against c19's 0.28 s, stable across runs.

**Of the tests**: the `check_index` + `checksum_verify` suite is **109 tests in
~34 s** (c19: 87 tests, ~50 s). The twenty-five new tests add ~4 s in total; the
two new corruption sweeps are deliberately sampled (a fixed stride giving
47 and 43 corruptions) rather than exhaustive, because the number they assert
is a *rate* and a rate does not need every byte.

---

## Gates

- `cargo fmt --all` — clean.
- `cargo clippy -p lucene-codecs --all-targets -- -D warnings` — clean (this
  is the crate F1/F2 are in).
- `cargo clippy -p lucene-index --all-targets -- -D warnings` — **clean for
  this batch's files**. The command itself is red, on exactly one diagnostic,
  every time it was run during this batch: `unused variable: doc_order` at
  `crates/lucene-index/src/merge.rs:3677`, which is c22's. Zero diagnostics
  anywhere in the run name `check_index.rs` or `checksum_verify.rs`.
  (It was also blocked for part of the batch by c24's in-flight arithmetic
  audit making `lucene-codecs` fail to compile under clippy — `lz4.rs`, then
  `regexp.rs`, markers removed ahead of the fixes. That has since cleared.)
- `cargo test -p lucene-index --lib -- check_index checksum_verify` — **109
  passed, 0 failed**, including
  `every_real_lucene_index_fixture_passes_every_check`.
- `cargo test -p lucene-index` (whole crate, F0 changes a public type) —
  **643 lib tests green** on the last clean run of the tree; at the very end
  one unrelated failure appeared in another batch's in-flight work (see
  Handoffs). Every integration binary is green, including c23's new
  `positions_write_path.rs` — which drives this module's positional and offset
  ordering checks from writer-produced input for the first time — and
  `index_sort_fixtures.rs`, the one external caller of the `Check` type F0
  changed.
- `cargo test -p lucene-codecs` — **all green** (1 145 lib tests + 32 test
  binaries), which is what covers F1/F2's blast radius.
- `python3 scripts/check-parity.py` — ok.
- `python3 scripts/check-arith-allows.py` — **ok after this batch fixed it**.
  It was failing for everyone when this batch started its final pass: the
  burn-down table in `docs/arithmetic-gate.md` claimed 11 unaudited modules in
  `lucene-index` and 12 in `lucene-codecs` against a tree holding 3 and 5,
  because c24 has been ticking modules off without updating the table.
  Corrected to the tree, script green. It went red again minutes later on
  **seven `#[allow]`s in c24's in-flight `term_vectors.rs` audit** — their
  code, their proofs (`ARITH (both partial-chunk walks):` and `ARITH as
  above:` do not match the script's required `ARITH:` marker; two have no
  proof block at all). Nothing this batch added uses `#[allow]` in that file.
  See Handoffs.
- `scripts/verify-write-path.sh` — **22/22**, confirmed by running it, not
  assumed. It was 21/21 at the start of this batch; a concurrent batch (c23)
  added `VerifyPositionsSegment <- write_positions_segment_fixture` while this
  one ran.

### Verdict

**`crates/lucene-index/src/check_index.rs`** — swept clean, requirement met.
94.19% → **97.25%**, over the ≥95% bar. 45 previously-unfired check arms are
now driven by a test that names them; 10 more are gone, each with the proof
that it could not fire left at the site. ~39 arms remain, every one of them
listed above with the fixture shape that would close it — the largest block
(`doc_values`, 30 lines) has a named, shaped next step (`patch_dvm`). And the module's result type no longer lets a check that could
not run read as one that passed — which is the finding this batch would have
been worth writing up even if the number had not moved.

**`crates/lucene-index/src/checksum_verify.rs`** — untouched; already swept
clean by c19 at 97.11%.

**`crates/lucene-codecs/src/term_vectors.rs`** — *not* swept (it is c24's);
two CORRECTNESS defects fixed in `build_field`/`read_chunk` with a test that
aborts the test binary without the fix, and a carry-over asking c24 to audit
the rest of the file for the same class.

---

## Handoffs

- **c24 — `crates/lucene-codecs/src/term_vectors.rs`**: F1 and F2 are edits to
  a file c24 owns. Both are contained (one `get(..)`, one length guard, one
  `bounded()` helper in `build_field`) and both are covered by
  `lucene-codecs`' own lib tests plus the new `.tvd` sweep. **c24 has since
  built on them** — the LZ4 ceiling now sits above their `sum_byte_lengths`,
  which rejects a negative length outright rather than clamping it, and their
  `ARITH` proof for the per-document cursor loop rests on the bound F2
  introduced. Nothing was clobbered in either direction; A3's remaining sites
  in `read_chunk` (`total_fields`, `total_distinct_fields`) are theirs to
  finish and are named in the carry-over.
- **The arithmetic-gate burn-down counts** in `docs/arithmetic-gate.md` were
  stale (11/12 against a tree holding 3/5), so `scripts/check-arith-allows.py`
  was failing for everyone. Corrected to the tree. The number moves every time
  c24 ticks a module off, so whoever finishes last should re-run the script.
- **c22 — `crates/lucene-index/src/merge.rs:3677`**: an unused `doc_order`
  parameter. It is the **only** diagnostic keeping
  `cargo clippy -p lucene-index --all-targets -- -D warnings` red, and it is
  not in this batch's files.
- **Callers of `Check`**: F0 replaced the public `Check::passed` *field* with a
  `passed()` method plus a public `outcome: Outcome`. The only caller outside
  this module was `crates/lucene-index/tests/index_sort_fixtures.rs:177`,
  updated in place. Anything downstream that pattern-matches `Check` needs the
  same one-character change — and should consider whether it wants to treat
  `Outcome::Skipped` as a failure (it is not a pass) or report it separately.
- **c24 — `crates/lucene-codecs/src/term_vectors.rs`, seven `#[allow]`s**
  at 859, 1037, 1562, 1657, 1808, 1812 and 1920 that
  `scripts/check-arith-allows.py` rejects: four have no `ARITH:` proof block,
  and three write `ARITH (…)` or `ARITH as above:`, which the script's literal
  `ARITH:` match does not accept. All are c24's own lines from their audit of
  that module; this batch added no `#[allow]` there. Either write the proofs
  or relax the script's marker to `ARITH`.
- **c23(?) — `crates/lucene-index/src/index_writer.rs:13746`**: `index out of
  bounds: the len is 0 but the index is 0` in
  `a_segment_with_a_doc_values_update_merges_at_its_newest_generation`,
  appearing in the last minutes of this batch. Not `check_index`; `Outcome`
  is not used in that file.
- **c23 — `crates/lucene-index/src/index_writer.rs:8971`**: one `--lib` test
  failure (`commit_with_term_vector_field_writes_readable_term_vectors_for_multiple_docs_and_terms`,
  `assert!(doc0.fields[0].has_positions)`) from that batch's in-flight change
  to which term-vector axes `IndexWriter` emits. Not this batch's files; every
  `check_index`/`checksum_verify` test passes.

`cargo fmt --all` (the gate command) reformatted a handful of files owned by
other in-flight batches. That is formatting only.

## Carry-over

- [ ] **`patch_dvm`**, in the shape of this batch's `patch_tmd`: 30 of the 103
      remaining uncovered lines are doc-values arms that need a `.dvm` which
      decodes but disagrees with its `.dvd`. This is the single clearest next
      block and it is worth more than its line count: the ordinal-space arms
      are the ones that catch a dictionary nothing can ever match.
- [ ] **A byte-encoded vector fixture.** Every `.vec` in the repo is
      `Float32`, so `check_vectors`' whole `VectorEncoding::Byte` arm — 11
      lines of decode and length validation — has never run against anything.
      That is a *format* gap as much as a coverage one: c5's byte-vector write
      path has no differential test either.
- [ ] **A multi-index-dimension points fixture through this port's writer.**
      `points.leaf_bounds_subset_of_field` is skipped entirely for
      single-dimension fields, which is every hand-built points fixture here.
- [ ] **`connected_nodes_on_level`'s entry-point guard** (see the table): prove
      whether `read_field_entry` can produce `size == 0` with a non-zero
      `vectorIndexLength`. If it cannot, the guard joins D1–D10.
- [ ] **A `.tip` that disagrees with its `.tim`.** Eight arms (seek, re-seek,
      intersect) are unreachable by byte corruption and need a hand-built trie.
      c9 already noted that `postings.intersect_agrees` compares two
      implementations that share a decoder; this is where that gets tested.
- [ ] **c24 should re-audit `crates/lucene-codecs/src/term_vectors.rs` for the
      *indexing and allocation* half of the class.** F1 and F2 were in a module
      the arithmetic gate has already been through, because the gate covers
      arithmetic and shifts and not `buf[i]` or `Vec::with_capacity(n)` —
      exactly the two rows `docs/arithmetic-gate.md` records as uncovered.
      `build_field`'s siblings (`read_chunk`'s `term_offsets`/`field_offsets`
      slicing) are the same shape and were not reached by this batch's sweep.
