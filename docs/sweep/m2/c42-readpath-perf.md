# c42-readpath-perf

Sweep batch over the remaining takeable read-path performance items:
**14** (`DirectoryReader::open`), **18** (impacts computed against norm 1),
**21** (splitting term iteration from stats in `TermsEnum`) and **24**
(`StoredFieldsReader::document` materialising a whole `Document`).

Java read from the pinned tree, `/home/tuong/work/lucene-10.5.0`.

**All four closed.** One additional defect was found and fixed on the way, in
the measurement tooling itself (see §0), and it is the batch's most
transferable finding.

Measurement method throughout: **alternating min-of-N, both arms in one
process**, never criterion (which reported 83/91/129 µs for identical code on
this host -- `c24-arith-codecs`). Every "before" was re-run against the same
build as its "after"; where an arm could not exist in one process, it is said
so explicitly.

---

## 0. The measurement harness was reporting a stale binary

`scripts/bench-micro.sh` builds `benchmarks/rust-runner` and then ran
`benchmarks/rust-runner/target/release/$RUST_BIN` by hardcoded path.
`scripts/docker-test.sh` exports `CARGO_TARGET_DIR=/work/target-docker`, so
inside the container **the build landed there and the script ran whatever host
binary happened to be sitting in the bind-mounted `target/` directory** -- in
this case one from 2026-08-29.

| | reported | actual |
|---|---|---|
| `micro reader_open`, Rust | 438 158 ns | 88 859 ns |
| ratio vs Lucene | 0.79x (Rust slower) | **4.31x (Rust faster)** |

It also reported *the same figure to three digits* (438158.688 then
438149.694 ns) across a change that moved the operation by 20% -- which is
what made it visible at all.

**Fixed** in `scripts/bench-micro.sh` and `scripts/bench-compare.sh`: both now
resolve `${CARGO_TARGET_DIR:-<the crate's own target>}`.

`docs/mechanical-gates.md`'s rule -- "a gate nobody has seen fail is a gate
nobody should trust" -- applies to a *measurement* the same way. What this one
could not catch, and still cannot, is a binary that is stale for some other
reason (an interrupted build, a `--offline` failure the `--quiet` build
swallowed). Making it print the binary's mtime would close that; not done.

---

## 1. Item 14 -- `DirectoryReader::open`

**Java counterparts**: `index/DirectoryReader.open`,
`index/StandardDirectoryReader`, `index/SegmentCoreReaders`,
`store/MMapDirectory` + `store/MemorySegmentIndexInputProvider`.

**Rust**: `crates/lucene-search/src/directory_reader.rs`,
`crates/lucene-store/src/directory.rs`.

### The premise was stale by ~550x

The ledger carried **52.7 ms, ~155x Lucene** from `verdict-m1.6.md`. That
predates c1 (lazy blocktree), c12 (`open_shared`, 579 → 120.7 µs) and b1's
mmap change (which removed a 1.57 GB per-open copy). The instruction was to
re-measure before planning, and re-measuring is the whole of finding 1.

### Profile, before changing anything

`crates/lucene-search/examples/reader_open_profile.rs` replays
`SegmentReader::open`'s phases one at a time through the same public functions,
over `benchmarks/.corpus/merged` (one segment, 5 M documents, 1.6 GB of
postings). The phase sum is printed beside the real `DirectoryReader::open` so
a drifted replica is visible rather than assumed.

Min of 60, everything mapped (i.e. the pre-batch behaviour):

| phase | µs |
|---|---|
| `segment_infos::read_latest` (`list_all` + parse `segments_N`) | 10.71 |
| `.si` open + parse | 6.22 |
| `.fnm` open + parse | 5.60 |
| `.tim`/`.tip`/`.tmd` mmap only | 6.10 |
| **`blocktree::open_shared` (mappings held)** | **0.37** |
| `.doc` mmap | 2.79 |
| `.pos` mmap | 2.75 |
| `.pay` mmap | 2.76 |
| `.kdm` / `.kdi` / `.kdd` mmap | 1.86 / 1.86 / 2.35 |
| `.dvm`/`.dvd` open + `parse_meta` | 7.82 |
| `.nvm`/`.nvd` open + `parse_meta` + validate + footer | 12.07 |
| **`open_segments`: `Doc`/`Pos`/`PayInput::open`** | **0.14** |
| phase sum | 63.68 |
| `DirectoryReader::open` (measured) | 81.97 |
| `open` + `open_segments` (measured) | 93.90 |

**The answer: it is entirely file opening.** Every decode is under a
microsecond -- the term dictionary is 0.37 µs, all three postings input headers
together are 0.14 µs, `parse_meta` for doc values and norms are sub-µs inside
their rows. What is left is fourteen `dir.open` calls at 1.2-2.8 µs each, plus
the first page fault each parsed file takes.

Against real Lucene on the same index, through `scripts/bench-micro.sh`
(once §0 was fixed): **88.9 µs vs 383 µs, 4.31x faster**. Note the Rust arm
does `open` + `open_segments` while Java's `DirectoryReader.open` additionally
opens stored fields, term vectors, points and per-field producers -- so Lucene
does strictly more work, and the ratio flatters this port by an unknown
amount. It is nevertheless nowhere near 155x, in either direction.

### [PERF] Finding 14.1 -- mapping a 642-byte file costs more than reading it

**Java**: `MMapDirectory.openInput` maps unconditionally; the only thing the
size influences is chunking (`chunkSizePower`, default 16 GB) and which arena
the mapping joins (`RefCountedSharedArena`, grouped per segment).

**We**: same, until this batch -- `Input` already had an `Owned` variant,
because `FsDirectory` produces it, but `MmapDirectory` never used it.

**Measured** (min of 60, same file, same process):

| file | `open` + `mmap` | `open` + `read` |
|---|---|---|
| `_1z.si`, 642 B | 1.86 µs | 1.19 µs |
| `_1z.fnm`, 542 B | 1.85 µs | 1.18 µs |
| `_1z_Lucene90_0.dvm`, 359 B | 1.87 µs | 1.19 µs |
| `_1z.nvm`, 139 B | 1.87 µs | 1.19 µs |
| `_1z_Lucene104_0.tmd`, 250 B | 1.86 µs | 1.19 µs |

And that is *before* the mapping's first page fault, which every one of these
files takes immediately because it is parsed whole at open.

**Resolution -- fixed** in `lucene_store::directory`: a file of at most
`SMALL_FILE_READ_THRESHOLD` (**16 KiB**) is `fs::read` instead of mapped.
16 KiB and not larger because the files a reader *holds* and then seeks in --
`.tip`, `.kdi` -- are the ones a copy would pessimise, and they are above it
on any index where it matters; the worst case the threshold can cost is one
16 KiB `memcpy`. `MmapDirectory::with_read_threshold(root, 0)` restores the
old behaviour, which is both the A/B arm and an escape hatch.

**Measured, alternating in one process, min of 60 / mean of 200:**

| | everything mapped | small files read |
|---|---|---|
| `DirectoryReader::open` (min) | 61.86 µs | **48.84 µs** (1.27x) |
| `DirectoryReader::open` (mean) | 75.71 µs | **55.15 µs** (1.37x) |
| `open` + `open_segments` (min) | 71.13 µs | **56.47 µs** (1.26x) |

**Memory axis**: the copies are bounded by 16 KiB per file and 7 such files per
segment -- ~5 KB in practice on this corpus, and each one is dropped as soon as
its parse finishes for every file except `.tmd`/`.nvd`/`.dvd`, which the reader
held anyway. It is a strict reduction in *address space*, since 7 mappings are
no longer created.

This is a deliberate divergence from Java, recorded in `docs/parity.md`.

### Recorded, not fixed

- **The three big postings mappings are 8.4 µs of the remaining 48.8 µs**, and
  they dominate the min-vs-mean spread (48.8 vs 55.1): mapping and unmapping
  1.6 GB has a variable tail. Lucene amortises the equivalent by grouping a
  segment's files into one `RefCountedSharedArena`
  (`MemorySegmentIndexInputProvider.getSharedArena`, keyed by segment name), so
  the segment's mappings are torn down together. This port has no equivalent
  and would need one to go further. **Sized, not attempted.**
- **`segment_infos::read_latest` is 10.7 µs → 7.7 µs**, of which most is
  `list_all` (a `readdir` of 23 entries, allocating a `String` each, then a
  sort). Java's `SegmentInfos.readLatestCommit` lists too. Not worth touching
  at this size; would matter on a directory with thousands of files.
- **`.kdm`/`.kdi`/`.kdd` are mapped even for a reader that never runs a points
  query** (6.1 µs). Java is eager here too (`SegmentCoreReaders` constructs the
  `PointsReader` when `fieldInfos.hasPointValues()`), so this is parity, not a
  gap -- but it is the largest single thing a lazy-open design would remove.

### Verdict

Closed. The item's headline number was wrong by ~550x, the diagnosis
("dominated by everything else") is now precise (**it is the `open` syscall,
and nothing else**), and the one contained lever was taken and measured on both
axes.

---

## 2. Item 18 -- impacts computed against norm 1

**Java counterparts**: `codecs/CompetitiveImpactAccumulator`,
`codecs/lucene104/Lucene104PostingsWriter.{startDoc,flushDocBlock,writeLevel1SkipData,writeImpacts}`,
`index/CheckIndex.checkImpacts`.

**Rust**: `crates/lucene-codecs/src/postings_writer.rs`,
`crates/lucene-index/src/{index_writer.rs,merge.rs}`.

### Method correspondence

| Java | Rust | state |
|---|---|---|
| `CompetitiveImpactAccumulator.add(int, long)` | `CompetitiveImpactAccumulator::add` | identical, including the `-128..=127` table / overflow-set split |
| `CompetitiveImpactAccumulator.getCompetitiveFreqNormPairs()` | `::get_competitive_freq_norm_pairs` | identical |
| `CompetitiveImpactAccumulator.add(Impact, TreeSet)` | `::add_to_set` | identical (`ceiling` → `partition_point`, `headSet(..).descendingIterator()` → a backwards walk) |
| `CompetitiveImpactAccumulator.addAll/copy/clear` | — | **not-in-Rust, deliberately**: this writer builds each file whole rather than streaming, so a level-1 span accumulates over its 8192 documents directly. The frontier of a union is the frontier of the union of frontiers, so the result is the same set Java's `addAll` produces. |
| `Lucene104PostingsWriter.writeImpacts` | `write_impacts` | identical |
| `Lucene104PostingsWriter.startDoc`'s norm lookup | `norm_of` + `competitive_impacts` | identical, both fallbacks included (`fieldHasNorms == false` and `advanceExact == false` → `1L`) |

### [PERF, and a soundness hazard] Finding 18.1

**Java** feeds `NormsProducer.getNorms(fieldInfo)` per document into
`CompetitiveImpactAccumulator` and writes the resulting competitive frontier
as each block's and span's impacts.

**We** wrote a single `(maxFreq, 1)` pair, because `FieldPostingsInput`
carried no norms. Sound -- norm 1 is the shortest field and so the
highest-scoring -- but loose enough that MAXSCORE almost never prunes on a
normed field.

**Consequence of getting it wrong in the other direction**: an impact below a
real document's score makes MAXSCORE skip a block containing a hit. Not a
score difference -- a **missing hit**, silently.

**Resolution -- fixed**:

1. `CompetitiveImpactAccumulator` ported, including the `otherFreqNormPairs`
   overflow set (reachable from a merge of a segment whose `.nvd` stores more
   than one byte per norm) and Java's *unsigned* norm ordering, which is what
   makes a sign-extended byte 200 sort after byte 100.
2. `postings_writer::write_fields_with_norms(inputs, &[FieldNorms], ...)` --
   `FieldNorms` is one field's dense per-doc column, which is what
   `NumericDocValues.advanceExact`/`longValue` amounts to for a writer.
   `write_fields` (no norms) is kept and is **exactly** Java's
   `fieldHasNorms == false` path, so every existing byte-level test is
   unchanged and passes unchanged.
3. Both production callers wired: `IndexWriter::build_norms_output` now returns
   its per-field columns (it computes them anyway, one flush earlier than the
   postings), and `merge.rs` hands the postings writer the same merged norms it
   writes to `.nvd`.

**Why the frontier and not a corner.** `(maxFreq, minNorm)` would also be
sound, and is what a shortcut reaches for -- but it is looser than Lucene's and
need not correspond to any real document. `(maxFreq, maxNorm)` is the shortcut
that *looks* right and is **unsound**; the negative control below is exactly
that.

### The test that would catch a skipped document

`crates/lucene-search/tests/impacts_soundness.rs`, over postings this port
writes and reads back through the real decoder (10 240 documents: one full
level-1 span plus eight further full blocks, freq and norm varying together
within each block so the frontier is several pairs, and norm bytes running past
127 so the sign-extended/unsigned path is exercised rather than assumed):

1. `bounds_never_fall_below_a_real_document_score` -- for every level-0 block
   **and** every level-1 span, `max_score_for_impacts` is `>=` the BM25 score
   of every document it covers. This is the invariant a skipped document
   violates, asserted at its source.
2. `pruned_top_n_equals_brute_force_top_n` -- a MAXSCORE-shaped block skip
   driven by those bounds returns the same top-`n` (1, 10, 100) as scoring
   every document. It also asserts that blocks *were* skipped, so it cannot
   pass vacuously.
3. `real_norms_make_the_bound_tighter_than_norm_one` -- the win: the same
   postings written without norms give a strictly higher bound, for at least
   one block and never a lower one.

**Verified to fail** by replacing `competitive_impacts` with the
`(maxFreq, maxNorm)` corner: (1) trips with
`doc 4 scores 4.4615026, above its block's bound 4.4371967`, and (2) prints the
missing hits directly --

```
left:  [doc 4, doc 24, doc 44, doc 64, doc 84, ...]   (brute force)
right: [doc 4, doc 24, doc 9,  doc 29, doc 8,  ...]   (pruned)
```

`+ 8` unit tests on the accumulator itself, including the wide-norm overflow
branch, the byte/wide merge, unsigned ordering of sign-extended norms, and a
`write_impacts` → `decode_impacts` round trip through the *read* side.

### And real Lucene now validates it

`CheckIndex.checkImpacts` requires non-empty impacts, a non-zero **first**
norm, and strictly increasing freq *and* unsigned norm. Every one of those
rules was vacuous while each block carried a single pair -- so
`verify-write-path.sh` passing 23/23 proved nothing about the new output.

`write_full_segment_fixture` now repeats `shared` `1 + (i % 4)` times, so its
frequency and its document's length vary together and `shared`'s blocks carry a
genuine multi-entry frontier; `VerifyFullSegment` gained
`checkImpactsHaveSeveralEntries`, which walks that term through Java's own
`ImpactsEnum` and fails if the richest impacts list has fewer than two entries.
**Verified to fail** by flattening the fixture back to `repeat = 1`:

```
FAIL (java verify)  VerifyFullSegment <- write_full_segment_fixture
    MISMATCH body:shared's richest impacts list had 1 entry/entries -- the
    fixture no longer exercises multi-impact blocks, so CheckIndex's impact
    rules are being checked against nothing
```

`verify-write-path.sh` is **23/23** with the fixture restored.

### [INTENTIONAL] Finding 18.2 -- norm 0

Java asserts `norm != 0` in `startDoc`; `CheckIndex` rejects a first impact
with norm 0. Norm 0 decodes to field length 0, which out-scores *every* other
norm -- so an impact carrying it is a sound bound but an illegal byte. It is
unreachable by construction here (a norm of 0 is what a document carrying the
field with **no tokens** gets, and such a document has no posting), and the
port carries Java's assertion as a `debug_assert!` at the same place. Mapping
0 → 1 would have been *unsound*, which is why it is not done.

### Verdict

Closed, done properly. c8's warning ("real norms need
`CompetitiveImpactAccumulator` in the same change, or the bounds become
unsound") was correct and was honoured rather than worked around.

---

## 3. Item 21 -- split term iteration from stats in `TermsEnum`

**Java counterparts**: `codecs/lucene103/blocktree/SegmentTermsEnum.{next,term,docFreq,totalTermFreq}`,
`SegmentTermsEnumFrame.decodeMetaData`, `IntersectTermsEnum.next`.

**Rust**: `crates/lucene-codecs/src/blocktree.rs`.

### [PERF] Finding 21.1

**Java**: `next()` decodes only the term bytes; `decodeMetaData` is deferred to
whichever of `docFreq()`/`totalTermFreq()`/`postings()` asks first, and is
memoised per frame position by `metaDataUpto`. `IntersectTermsEnum.next()`
never asks.

**We**: `try_next()` returned `(&[u8], TermStats)` and so always ran
`decode_meta_data` -- a stats vint *and* a full `TermMetadata` decode
(`.doc`/`.pos`/`.pay` file pointers, singleton pulsing) -- for every term,
including every term a scan rejects.

**Resolution -- fixed**: `TermsEnum` gains `try_next_term()` (Java's `next()`),
`term()` (Java's `term()`) and `try_stats()` (Java's
`docFreq()`/`totalTermFreq()`). `try_stats` is memoised by the same
`meta_data_upto` Java uses, so `try_next_term()` then `try_stats()` costs
exactly what the fused `try_next()` costs -- the saving is real only for the
terms whose stats are never asked for, which is the honest shape. `try_next()`
survives for callers that want both, with a doc comment saying which one is
Java's.

**Measured** (`crates/lucene-search/examples/terms_enum_split.rs`, alternating
min-of-N, both arms in one process, `body` field of `benchmarks/.corpus/merged`,
200 000 terms):

| arm | total | per term |
|---|---|---|
| `try_next` (term + stats) | 4.581 ms | 22.90 ns |
| `try_next_term` (term only) | 2.084 ms | **10.42 ns** |
| | | **2.20x** |

The ledger's recorded comparison was 27 ns/term against Lucene's 20.5 ns. A
bytes-only walk is now roughly **half** Lucene's figure; the fused walk, at
22.90 ns, is within 12% of it.

**Call sites migrated** (three, all production):

- `blocktree::Intersect::next_result` -- the wildcard/prefix/regexp/fuzzy term
  expansion, which was decoding metadata for **every rejected term**.
- `check_index::compare_intersect_with_scan` -- the linear scan half, which
  compares term *bytes*.
- `IndexWriter::resolve_term_span` -- the term-range delete walk, which
  classifies on bytes and re-seeks the terms it keeps anyway.

The recorded ripple ("changes `check_index`'s and the intersect iterators' call
shape") was accurate but smaller than it sounded; c39 had already paid half of
it for a different reason.

### [INTENTIONAL] Finding 21.2 -- one behavioural consequence

A corrupt stats blob on a term the intersection *rejects* is no longer
surfaced as an error, because the walk no longer decodes it. This matches Java
exactly: `IntersectTermsEnum.next()` does not call `decodeMetaData` either, so
neither engine sees the corruption until something asks for that term's stats.
Terms the intersection *yields* still decode, and still report.

### Verdict

Closed.

---

## 4. Item 24 -- `StoredFieldsReader::document()` materialises a whole `Document`

**Java counterparts**: `index/StoredFieldVisitor` (+ `.Status`),
`document/DocumentStoredFieldVisitor`,
`codecs/lucene90/compressing/Lucene90CompressingStoredFieldsReader.{document,readField,skipField}`.

**Rust**: `crates/lucene-codecs/src/stored_fields.rs`.

### Method correspondence

| Java | Rust | state |
|---|---|---|
| `StoredFieldVisitor.needsField(FieldInfo)` | `StoredFieldVisitor::needs_field(i32)` | identical, by field **number** (see below) |
| `StoredFieldVisitor.{string,binary,int,long,float,double}Field` | `::{string,binary,int,long,float,double}_field` | identical, all with no-op defaults as Java's non-abstract methods have; values are **borrowed** |
| `StoredFieldVisitor.Status` | `VisitStatus` | identical |
| `DocumentStoredFieldVisitor` | `DocumentVisitor::{all,for_fields}` | identical, including "answer `NO`, never `STOP`, even once every wanted field is found" (a document's fields are in no defined order) |
| `Lucene90CompressingStoredFieldsReader.document(int, StoredFieldVisitor)` | `StoredFieldsReader::visit_document` + `stored_fields::visit_document` | identical, including the "don't `skipField` on the last field value; treat like STOP" shortcut |
| `...readField(DataInput, visitor, FieldInfo, int)` | `visit_field` | identical |
| `...skipField(DataInput, int)` | `skip_field` | identical |
| `...document(int)` (the `DocumentStoredFieldVisitor` convenience) | `StoredFieldsReader::document` / `parse_document` | now **implemented as** the visitor loop with a `DocumentVisitor::all`, so there is one field-decode path; the old `read_field` is deleted |

### [MISSING] Finding 24.1

**Java** hands the caller one field at a time and skips the value bytes of a
field the caller declines, so retrieving one field of a wide document costs
that field plus a length vint per other field.

**We** decoded and allocated every field of the document unconditionally.

**Resolution -- fixed** as above. Values reach the visitor borrowed
(`&str`/`&[u8]`, the latter sliced straight out of the decompressed buffer
rather than copied into a `Vec` first), so a visitor that keeps nothing
allocates nothing.

**Measured** (`crates/lucene-codecs/examples/stored_fields_visitor.rs`, 4096
documents read on a stride so consecutive reads land in different chunks,
retrieving the **last** field so every other field must actually be skipped;
min of N, both arms in one process):

| fields per document | `document()` | visit one field | |
|---|---|---|---|
| 1 | 2.533 µs/doc | 2.540 µs/doc | 1.00x |
| 4 | 1.825 | 1.764 | 1.03x |
| 16 | 2.213 | 1.779 | **1.24x** |
| 64 | 5.025 | 2.785 | **1.80x** |

1.00x at one field is the no-regression check: there is nothing to skip, so
the visitor must cost what the old path cost.

**What the visitor does not save, and this is most of the residue**: the chunk
still has to be decompressed over the document's byte range
(`serialized_document`), and that range is still copied into a `Vec`. Java has
the same decompression but hands the visitor a `DataInput` over its own block
buffer rather than a copy. Removing the copy would need `serialized_document`
to return a borrow of a cached chunk, which is `ChunkCursor`'s shape and a
larger change. **Recorded, not attempted.**

### [INTENTIONAL] Finding 24.2 -- fields are numbers, not `FieldInfo`s

Java's `needsField` takes a `FieldInfo`. This reader decodes `.fdt` alone and
has never been handed a `FieldInfos` (see `stored_fields::open`'s signature);
the field *number* is what the wire format carries, and a caller that wants
names already holds the `.fnm` mapping `field_infos` produced. Passing a
`FieldInfos` in would be a new dependency on this decoder purely to re-derive
something the caller has.

### Verdict

Closed.

---

## Cross-cutting notes

### Regressions checked, none found

- **c1's segment open** -- `blocktree::open_shared` is 0.37 µs in the profile
  above, well under c1's 0.175 ms.
- **c39's jump-table seeks** and **c20's highlight** -- untouched code paths;
  the full suite passes.
- **c38's memory numbers** -- untouched. Two memory changes, both accounted:
  `SMALL_FILE_READ_THRESHOLD`'s bounded ≤16 KiB copies (§1, both axes), and
  the norm columns item 18 needs. The latter went through review: the first
  version *cloned* the dense column, adding a retained `8 × maxDoc` per normed
  field across exactly the window `build_and_write_segment`'s
  norms-then-postings ordering exists to keep small, and made three copies of
  it on the merge path. It now builds **one** dense-by-doc-id column per
  field, which `norms::NormsField::Dense` borrows and the postings writer then
  takes by move -- so the flush path allocates no more than before item 18,
  and the merge path allocates one `Vec<i64>` *fewer* (`dense_columns` is
  gone).
- **Byte-for-byte output**: `write_fields` without norms writes exactly what it
  wrote before (the frontier of a constant-norm block *is* one `(maxFreq, 1)`
  pair), which is why 1271 `lucene-codecs` tests passed unchanged after the
  impacts change.

### What a future batch should look at

- **A per-segment arena for mappings** (`RefCountedSharedArena`), item 14's
  residue: 8.4 µs of the remaining 48.8 µs is mapping and unmapping 1.6 GB of
  postings, and it is the whole of the min-vs-mean spread.
- **`serialized_document`'s copy**, item 24's residue: the visitor removes the
  per-field allocations but not the per-document one.
- **`check_index` does not validate impacts** the way `CheckIndex.checkImpacts`
  does. This port's own `CheckIndex`-equivalent walks postings but does not
  assert the impacts' ordering, non-emptiness or `freq <= blockMax`. Real
  Lucene now checks this port's output (§2), so the gap is in *this port's
  checker*, not in the bytes. Not opened as a numbered item; noted here.

---

## Gate

`scripts/docker-test.sh gate` — **ok** (exit 0), re-run after the Tier-2
review's seven fixes landed (and twice before that).

- `cargo fmt --check`, `cargo clippy -D warnings` (x86_64 **and**
  aarch64), `cargo check` on `benchmarks/rust-runner`, `check-arith-allows`,
  `check-port-invariants` (ledger single-list included), `check-parity`,
  `check-java-refs`, `cargo doc` link lints: all pass.
- `cargo llvm-cov --workspace --fail-under-lines 95`: **98.12%** lines
  (was 98.14%). Every file this batch touched is above the per-file bar:

| file | lines |
|---|---|
| `lucene-codecs/src/postings_writer.rs` | 99.36% |
| `lucene-codecs/src/stored_fields.rs` | 97.18% |
| `lucene-codecs/src/fst.rs` | 97.53% |
| `lucene-codecs/src/blocktree.rs` | 96.64% |
| `lucene-index/src/index_writer.rs` | 98.34% |
| `lucene-index/src/merge.rs` | 98.60% |
| `lucene-search/src/directory_reader.rs` | 97.81% |
| `lucene-store/src/directory.rs` | 96.85% |

- `scripts/verify-write-path.sh`: **23/23**, with `VerifyFullSegment` now
  additionally reading this port's multi-impact blocks through Java's
  `ImpactsEnum` and `CheckIndex.checkImpacts`.

### Tier-2 review

The `quality-reviewer` subagent reviewed this diff. It **cleared the
soundness-critical parts** -- `add_to_set` against Java's private
`add(Impact, TreeSet)` (including that the `Vec`-vs-`TreeSet` duplicate-insert
divergence is unreachable), `get_competitive_freq_norm_pairs` with a non-empty
overflow set (Java's `new TreeSet<>(SortedSet)` preserves the comparator, so
`self.other.clone()` matches), the level-1 union, the byte-identical normless
path, and norm-column alignment on both the flush and merge paths -- and
raised seven findings. **All seven are fixed:**

1. **GATING.** Three tests that write a small file to an `MmapDirectory` and
   then assert something about the mapping had *silently stopped using one*:
   `fst.rs`'s `read_borrowed_over_a_real_mmap_directory_input`,
   `tests/fst_borrowed_seek_fixtures.rs`'s
   `seek_and_enumerate_over_a_real_mmap_directory_backed_borrowed_fst` (the
   only end-to-end coverage of a borrowed FST over a real OS mapping, over a
   78-byte fixture), and -- found while checking the reviewer's two --
   `directory_fixtures.rs`'s `mmap_backend_reads_same_bytes_as_fs_backend`,
   which was left comparing one `Input::Owned` against another. All three now
   use `with_read_threshold(_, 0)` **and assert the variant**, so the next
   threshold change cannot silently un-test them.
2. `MmapDirectory::open`'s small-file arm called `fs::read(&path)`, which
   re-opens by name and re-`fstat`s -- two `open`+`fstat` pairs in the arm
   whose whole justification is that it is cheaper than a mapping (and a
   TOCTOU window `FsDirectory` does not have). It now reads from the
   descriptor it already holds.
3. The norms clone, above.
4. `write_impacts`' first-entry `debug_assert` had an `|| prev == (0,0)`
   escape hatch, which disabled it for exactly the entry
   `CheckIndex.checkImpacts` rejects ("First impact had a norm == 0"). Java's
   assertion is unconditional; so is this one now.
5. Impacts non-emptiness became an implicit dependency on `validate_field`
   250 lines away when the `.max(1)` was replaced; `debug_assert!` at both
   levels now states it where it is relied on.
6. The `type Impact` alias's doc claimed a separation that does not exist.
   (Already reworded before the review landed.)
7. An `.expect()` on a decode path reachable from the JVM through a
   wildcard/regexp/fuzzy expansion, against AGENTS.md invariant 5. Now an
   `else` that reads as end-of-terms.

One reviewer claim did **not** survive testing, and is worth recording: the
report suggested `MmapDirectory` previously returned `Err` for a zero-length
file because `mmap(2)` rejects a zero length. `memmap2::Mmap::map`
special-cases it and returns an empty mapping, so the two backends already
agreed and the threshold changed nothing observable. The new test
(`a_zero_length_file_reads_as_empty_on_every_backend`) asserts what actually
happens on all three paths rather than what either of us predicted.

The reviewer's own verification, recorded here because it is the part a future
reader will want:

- `add_to_set` vs Java's private `add(Impact, TreeSet)`: the ordering key
  `(freq asc, unsigned norm desc)` is Java's comparator verbatim;
  `partition_point` at that key is `ceiling`; the backwards walk from the
  insertion point is `headSet(newEntry, false).descendingIterator()`. The one
  place a `Vec` could diverge from a `TreeSet` -- inserting a duplicate where
  `TreeSet.add` is a no-op -- is unreachable, because an element equal under
  the comparator is an identical impact and the `next.norm <=u newEntry.norm`
  branch has already returned.
- The highest-freq entry is always in the frontier (nothing dominates it, and
  `add_to_set` only removes entries whose freq is strictly lower), which is
  what keeps `max_score_for_impacts_unnormed` -- used when a leaf has no norms
  even though the writer had them -- sound as well.
- Norm-column alignment is cross-checked by real Lucene on the flush path:
  `VerifyFullSegment.checkNorms(reader, "body")` asserts every document's norm
  equals `SmallFloat.intToByte4(bodyLength(doc))`, and the impacts are built
  from that same column.
- `Input::Owned` where a caller previously got `Input::Mapped`: no consumer
  compares pointers or depends on the mapping's identity; `blocktree::open_shared`
  takes `Arc<dyn AsRef<[u8]>>` and accepts either.

Both the review's conclusions and these agree, which is the point of having
had both.
