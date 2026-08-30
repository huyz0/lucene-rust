# c18-version-audit — did the sweep read the wrong Lucene?

**Batch type:** correction audit, not a sweep. No broad refactoring.

## Why this batch exists

`AGENTS.md` pins Lucene **10.5.0**, and `scripts/lib-lucene-jars.sh` downloads
10.5.0 jars — so every fixture and verifier in this repo is ground truth for
10.5.0. But `/home/tuong/work/lucene`, the tree every previous sweep batch read
Java *source* from, is a checkout of `main` at 2026-08-26: 4574 commits and
1261 files / ~62k lines past the tag. Batch c14 found this independently for
`ReadersAndUpdates`. This batch checks the other 31 ported classes that differ
between `releases/lucene/10.5.0` and `main`.

Source of truth for this audit: `/home/tuong/work/lucene-10.5.0` (a `git archive`
of the tag). Diffs obtained with
`git -C /home/tuong/work/lucene diff releases/lucene/10.5.0..main -- <path>`.

## Verdict table

| Java class | Diff | Verdict | What actually differs |
|---|---|---|---|
| `CheckIndex` | 333/31 | **VERSION-DEFECT** (scope, non-correctness) | `main` adds `testHnswGraphs`, FLOAT16; 10.5.0 has neither. We ported `testHnswGraphs`. |
| `Lucene90DocValuesProducer` | 227/52 | CLEAN | Query-time SIMD `rangeIntoBitSet`/bulk `binaryValues` only. **No encoding change.** |
| `BooleanQuery` | 80/71 | **VERSION-DEFECT** (MISSING + wrong rationale) | `main` reroutes SHOULD/MUST dedup through `Similarity.computeQueryTermWeight`; 10.5.0's dedup is a pure structural boost-sum. |
| `BM25Similarity` | 74/30 | CLEAN | `main` adds `k3` (default `-1` = disabled) and renames `CollectionStatistics`→`FieldStats`. Scoring formula unchanged. |
| `SegmentInfos` | 70/42 | CLEAN | `main` bumps `VERSION_CURRENT` 10→11 and adds a per-segment `DocValuesOverlays` block. We write `VERSION_86 = 10`, no overlays. |
| `CodecUtil` | 42/5 | CLEAN | Merge-abort-aware `checksumEntireFile` overload only. |
| `DataInput` | 40/44 | **VERSION-DEFECT** (doc only) | `main` bounds `readVInt`/`readVLong` at `shift < 32`/`< 64`; 10.5.0's loops are unbounded. Our doc comment quoted `main`. |
| `Lucene103BlockTreeTermsWriter` | 31/37 | CLEAN | `suffix` → `suffixLength` rename + try/catch refactor. |
| `RegExp` | 24/47 | CLEAN | `main` removes `DEPRECATED_COMPLEMENT` (not in `RegExp.ALL` in 10.5.0 either), renames `CaseFolding`→`CaseExpansion`, changes `toString` escaping. Syntax table untouched. |
| `TieredMergePolicy` | 23/66 | **VERSION-DEFECT** (CORRECTNESS) | `main` deletes `maxMergeAtOnce` entirely. 10.5.0 has `maxMergeAtOnce = 10` as a hard per-merge cap and `mergeFactor = (int) min(maxMergeAtOnce, segsPerTier)`. |
| `HnswGraphBuilder` | 18/25 | CLEAN | Constructor collapse (`M` read off `hnsw.maxConn()`) + abort check. No constant/algorithm change. |
| `Lucene90DocValuesConsumer` | 18/22 | CLEAN | Exception-handling refactor only. |
| `BKDWriter` | 17/34 | CLEAN | `BKDMergeQueue` → `Comparator`; `IOUtils` refactor. |
| `BooleanScorer` | 9/38 | CLEAN | `HeadPriorityQueue`/`TailPriorityQueue` → `PriorityQueue.usingLessThan`. |
| `Lucene104PostingsWriter` | 9/15 | **VERSION-DEFECT** (CORRECTNESS, write path) | Dense-block encoding choice: 10.5.0 `numBitsNextBitsPerValue <= docRange`; `main` `<= numBitSetLongs * Long.SIZE`. |
| `Lucene103BlockTreeTermsReader` | 9/13 | CLEAN | `checkIntegrity(merge)` signature + refactor. |
| `Lucene104PostingsReader` | 9/11 | CLEAN | `checkIntegrity(merge)` + stale comment fixes. |
| `QueryParserBase` | 8/32 | CLEAN | `main` removes `determinizeWorkLimit`; we never had it. |
| `MultiPhraseQuery` | 8/25 | CLEAN | `PriorityQueue` refactor + `TermStatistics`→`TermStats` rename. |
| `PhraseQuery` | 7/10 | CLEAN | Javadoc + `TermStatistics`→`TermStats`. |
| `BKDReader` | 3/3 | CLEAN | Removed `throws IOException`. |
| `ForUtil` (lucene104) | 2/7 | CLEAN | `expand8` delegated to `VectorUtil`. Identical semantics, `BLOCK_SIZE == 256` in both. |
| `LZ4` | 2/2 | CLEAN | Two comment typos. |
| `DirectWriter` | 2/2 | CLEAN | Javadoc markup. |
| `OrdinalMap` | 2/13 | CLEAN | `TermsEnumPriorityQueue` → `PriorityQueue.usingComparator`. |
| `Lucene99FlatVectorsWriter` | 69/6 | CLEAN | Adds FLOAT16 (post-10.5.0 feature, not ported). |
| `Lucene99HnswVectorsWriter` | 32/18 | CLEAN | Same + refactor. |
| `IndexWriter` | 289/343 | CLEAN | `finally`/`success` → `catch (Throwable)`, `Event`→`IOConsumer`. |
| `IndexFileDeleter` | 4/12 | CLEAN | `catch (NumberFormatException _)`. |
| `Lucene90CompressingStoredFieldsWriter` | 5/11 | CLEAN | Refactor + `checkIntegrity(merge)`. |
| `Lucene90CompressingTermVectorsWriter` | 7/14 | CLEAN | Same. |
| `UnifiedHighlighter` | 12/12 | CLEAN | Typos + pattern-matching `instanceof`. |
| `Lucene90CompoundFormat` | 7/16 | **VERSION-DEFECT** (doc + unspecified tie order) | 10.5.0 drains a `PriorityQueue<SizedFile>`; `main` sorts a `List`. Our comment cited `main`. |

Also checked and clean: no other format-version constant changed anywhere under
`lucene/core/src/java/org/apache/lucene/{codecs,index}` between the tag and
`main` (`grep` over the whole diff for `VERSION_CURRENT|VERSION_*|CODEC_NAME|
*_EXTENSION`) — `SegmentInfos` is the only one, and we are on the 10.5.0 value.
`KnnSearchStrategy.DEFAULT_FILTERED_SEARCH_THRESHOLD` changed `0` → `60` in
`main`; not ported here.

---

## Findings

### 1. [CORRECTNESS] `TieredMergePolicy` lost Java's `maxMergeAtOnce` hard cap — FIXED

**Java (10.5.0)** `TieredMergePolicy` has two caps:

- `maxMergeAtOnce = 10` — a **hard** bound, `candidate.size() < maxMergeAtOnce`
  in `doFindMerges`' packing loop, applied for `NATURAL` *and*
  `FORCE_MERGE_DELETES`.
- `mergeFactor = (int) Math.min(maxMergeAtOnce, segsPerTier)` — the **soft**
  bound that a merge still under `floorSegmentBytes` is allowed to exceed, and
  the denominator of `score()`'s `skew` and of the `allowedSegCount` level walk.

**Java (`main`)** deletes `maxMergeAtOnce` outright: `mergeFactor` becomes
`(int) segsPerTier` and the hard bound is gone.

**We had** `main`'s shape: one knob (`max_merge_at_once`, defaulted to `8`),
`merge_factor() == max_merge_at_once`, no hard cap, and `skew = 1.0 /
segs_per_tier()`. The module doc even asserted "`maxMergeAtOnce` was removed
from `TieredMergePolicy` in Lucene 9", which is false for the pinned version.

**Consequence.** With the default config (`floorSegmentBytes = 16 MB`), a run of
sub-floor segments packs until it reaches the floor rather than stopping at ten
inputs. Real 10.5.0 caps every merge at 10 segments. Different merge grouping,
different segment topology.

**Does a fixture cover it?** `fixtures/data/merge_policy/` is generated by
`fixtures/src/GenMergePolicy.java` against the **real 10.5.0 jars**, so it *is*
correct ground truth — but it never reaches the cap. Its widest below-floor
scenario (`many_tiny_segments_below_floor`: 20 × 512 B, floor 4096) packs to
exactly 8 before `bytesThisMerge >= floorSegmentBytes` stops it. The 10-segment
bound is never exercised, which is why b10's line-by-line port passed.
Additionally the fixture test set `max_merge_at_once = segs_per_tier`, while
`GenMergePolicy.java` leaves Java's `maxMergeAtOnce` at its default of 10.

**Resolution — fixed** in `crates/lucene-index/src/merge_policy.rs`:

- `max_merge_at_once` default `8` → **`10`** (Java's field initialiser).
- `merge_factor()` → `min(max_merge_at_once, segs_per_tier).max(1)`.
- Added `candidate.len() < config.max_merge_at_once` to `do_find_merges`'
  packing loop.
- `score()`'s `skew` denominator: `segs_per_tier()` → `merge_factor()`.
- Module/field docs rewritten to describe 10.5.0.
- `crates/lucene-index/tests/merge_policy_fixtures.rs` now sets
  `max_merge_at_once: 10`, matching what the generator actually configured.
- Tests: `default_config_matches_real_lucene_defaults` now asserts
  `max_merge_at_once == 10` **and** `merge_factor() == 8`;
  `merge_factor_cap_respected_unless_below_floor` rebuilt around
  `maxMergeAtOnce=10, segsPerTier=4` (so the two caps are distinguishable) and
  asserts the hard cap still holds below the floor; new
  `max_merge_at_once_is_a_hard_cap_below_the_floor` pins a candidate packed to
  exactly 10.

### 2. [CORRECTNESS] `Lucene104PostingsWriter` used `main`'s dense-block encoding rule — FIXED

`flushDocBlock` decides between packed FOR and a unary bit set:

```java
// 10.5.0
} else if (numBitsNextBitsPerValue <= docRange) {
// main
} else if (numBitsNextBitsPerValue <= (numBitSetLongs * Long.SIZE)) {
```

`numBitSetLongs * 64 == ceil(docRange/64)*64 >= docRange`, so `main`'s rule is
strictly looser and picks packed FOR for every block in the band
`docRange < numBitsNext <= ceil(docRange/64)*64`.

`crates/lucene-codecs/src/postings_writer.rs` had `main`'s condition — and a
comment stating the port had been *changed* to it "until the M2 sweep", plus a
test (`full_block_encoding_choice_matches_lucene_in_the_disputed_band`)
asserting `main`'s chosen token. So an earlier sweep batch actively moved this
line onto `main`.

**Consequence.** Byte-level write-path divergence from real 10.5.0 for any
`.doc` block in that band. Both shapes are legal and both readers accept
either, so nothing breaks — but our `.doc` is not what 10.5.0 would have
written.

**Does a fixture cover it?** No. The postings fixtures are read-side (real
Lucene writes, we parse) and `scripts/verify-write-path.sh` opens our output
with real Lucene, which accepts either token. Only a token-level assertion
catches it.

**Resolution — fixed**: condition reverted to `num_bits_next_bits_per_value <=
doc_range`, comment rewritten to quote 10.5.0 and flag `main`'s later change as
a version trap, and the test's expectation flipped to `-12` (the 12-long bit
set) with the reasoning inverted.

### 3. [MISSING] `BooleanQuery` MUST/SHOULD dedup — the "cannot be ported" rationale is `main`'s, not 10.5.0's — **HANDOFF to `c16-knn-query`**

`crates/lucene-search/src/query.rs` (`Query::rewrite`) declines to implement
`BooleanQuery.rewrite`'s MUST/SHOULD duplicate folding, with a long rationale
that it is impossible without an `IndexSearcher`, because Java computes the
recombined boost from `IndexSearcher.getSimilarity().computeQueryTermWeight(count)`.

**That is `main`'s implementation.** In 10.5.0 the dedup is a pure structural
transform with no `Similarity` in sight: walk the `BoostQuery` chain, multiply
the boosts, sum per distinct query, and rebuild with one clause per query
carrying the summed boost (`if (boost != 1f) query = new BoostQuery(query, boost)`).
`computeQueryTermWeight` does not exist in 10.5.0's `Similarity` at all — it
arrived with `main`'s `k3` work.

**Consequence.** Scoring impact is small: BM25 is linear in the clause sum, so
`a a` scores the same whether it is two SHOULD clauses or one at boost 2 (modulo
float association for mixed boosts like `a^2 a^3`). What differs is query
*shape*: clause count (and therefore `maxClauseCount` pressure and scorer
count), `equals`/`hashCode`, and the explain tree. The port is not wrong — it
is *unjustified*: the stated reason for not doing it does not apply to 10.5.0.

**Resolution — recorded, not fixed.** `crates/lucene-search` is owned by
`c16-knn-query`. Precise handoff:

- File `crates/lucene-search/src/query.rs`, the doc block above
  `Query::rewrite` beginning "**Deliberately NOT implemented: `must`/`should`
  duplicate deduplication.**".
- The two bullets citing `computeQueryTermWeight` describe Lucene `main` and
  must be deleted.
- 10.5.0's actual rule, from
  `/home/tuong/work/lucene-10.5.0/lucene/core/src/java/org/apache/lucene/search/BooleanQuery.java`
  (`rewrite`, the two `Deduplicate ... clauses by summing up their boosts`
  blocks): gated on `clauseSets.get(SHOULD).size() > 0 && minimumNumberShouldMatch <= 1`
  for SHOULD and `clauseSets.get(MUST).size() > 0` for MUST; unwrap nested
  `BoostQuery` multiplying boosts; `shouldClauses.merge(query, boost, sum)`;
  rebuild only `if (map.size() != clauseSets.get(occur).size())`, preserving
  `minimumNumberShouldMatch` and re-adding all other-occur clauses unchanged.
- This is implementable in a pure structural `rewrite` with no `IndexSearcher`.

### 4. [MISSING/scope] `CheckIndex.testHnswGraphs` does not exist in 10.5.0 — **HANDOFF to `c15-postings-api`**

`crates/lucene-index/src/check_index.rs` implements `check_hnsw_graphs` as a
port of `CheckIndex.testHnswGraphs` (`testHnswGraph`, per-level node counts,
connectedness fractions). **`testHnswGraphs` was added after 10.5.0** — the tag's
`CheckIndex` has no such method, no `HnswGraphStatus`/`HnswGraphsStatus`, and no
`hnsw graphs` info-stream line.

**Consequence.** None for correctness: `CheckIndex` is a diagnostic, the extra
check is sound, and it only ever fails on genuinely broken graphs. But it is a
Rust-only extra relative to the pinned version, and the module doc presents it
as a Java counterpart. The same module's scope note says "the `Float16` vector
encoding (no counterpart in this port)" — 10.5.0's `VectorEncoding` has only
`BYTE` and `FLOAT32`; FLOAT16 is `main`'s.

**Resolution — recorded, not fixed** (file owned by `c15-postings-api`).
Suggested edit: relabel `check_hnsw_graphs` as a Rust-only extension with no
10.5.0 counterpart (or as a port of post-10.5.0 `CheckIndex`), and drop the
FLOAT16 sentence from the scope note.

### 5. [INTENTIONAL, doc] `DataInput::read_vint` documented `main`'s loop — FIXED

10.5.0:

```java
byte b = readByte(); int i = b & 0x7F;
for (int shift = 7; (b & 0x80) != 0; shift += 7) { b = readByte(); i |= (b & 0x7F) << shift; }
```

— **unbounded**: a corrupt run of `0xFF` reads until EOF, with `shift` wrapping
mod 32. `main` bounded it to `shift < 32` (and `< 64` for `readVLong`).

Our implementation's *shape* is 10.5.0's (do-while on the continuation bit) and
it stops at five bytes with `Error::MalformedVarint`; the doc comment and the
`vint_never_consumes_more_than_javas_five_bytes` test comment both described
`main`'s bounded loop as "what Java does". On well-formed input all three agree
byte for byte (`writeVInt` never sets the fifth continuation bit), so this is a
corrupt-input-only path and our erroring is a deliberate hardening over both.
Comments corrected in `crates/lucene-store/src/data_input.rs`.

`readMapOfStrings`/`readSetOfStrings` also changed (10.5.0: `TreeMap`/`TreeSet`
under 11 entries, `HashMap`/`HashSet` above; `main`: `Map.ofEntries`/`Set.of`).
Our port returns `Vec<(String, String)>` / `Vec<String>` in wire order, which is
round-trip-exact and independent of both — CLEAN, and strictly better than
either for a writer.

### 6. [INTENTIONAL, doc] `VectorEncoding` FLOAT16 — FIXED

10.5.0's `VectorEncoding` is `BYTE(1), FLOAT32(4)` — ordinals 0 and 1. `main`
appends `FLOAT16(2)` as ordinal 2. `read_vector_encoding` correctly rejects
ordinal 2, but its comment claimed "FLOAT16 == 2 exists in Lucene 10.5". Fixed
in `crates/lucene-codecs/src/vectors.rs` and the scope note in
`crates/lucene-codecs/src/hnsw_vectors.rs`. (The third instance is in
`check_index.rs` — see finding 4's handoff.)

### 7. [PERF/doc] `Lucene90CompoundFormat` sub-file ordering — comment FIXED

10.5.0 packs sub-files smallest-first by draining a `PriorityQueue<SizedFile>`;
`main` replaced it with `List.sort(Comparator.comparingLong(length))`. Our
`compound_format::write` uses Rust's stable `sort_by_key` (i.e. `main`'s shape)
and cited `Comparator.comparingLong(length)` in its doc.

For distinct lengths the orders coincide. For equal-length sub-files 10.5.0's
binary-heap pop order is unspecified, so `.cfs` packing order can differ from
real 10.5.0's. Nothing in the format depends on it — `.cfe` maps names to
`(offset, length)` and lookup is by name — and the compound-format fixture is
read-side only, so nothing pins it. Recorded in the doc comment rather than
reimplementing Lucene's heap; changing it would trade a deterministic stable
order for an unspecified one.

---

## Blast radius on the sweep's ~500 findings

**Small, and now bounded.** Of 32 classes that differ between 10.5.0 and `main`,
**26 differ only by refactoring, renames, javadoc, exception-handling idiom, or
features we do not port** — reading `main` for those produced the same
conclusions the tag would have. Four are real defects and two are documentation
defects.

The three highest-risk paths named in the brief all check out:

- **`BM25Similarity`** — b12's reasoning holds. `main`'s only semantic change is
  `k3`, which defaults to `-1` (disabled) and so scores identically; the `idf`,
  `tfNorm` and per-norm-byte cache are byte-identical between versions. The
  10.5.0 jars b12 scored against are ground truth and our port has no `k3`.
- **`TieredMergePolicy`** — b10's constants were right (`segsPerTier = 8.0`,
  `maxMergedSegmentBytes = 5 GiB`, `floorSegmentBytes = 16 MB`,
  `deletesPctAllowed = 20`, `forceMergeDeletesPctAllowed = 10`,
  `targetSearchConcurrency = 1`) and its fixture did use the 10.5.0 jars. What
  it missed was the *structure* `main` deleted: `maxMergeAtOnce`. Finding 1.
- **`RegExp`** — b8's syntax table is safe. `main`'s changes are the removal of
  a flag (`DEPRECATED_COMPLEMENT`, `0x10000`) that is not in `RegExp.ALL` in
  10.5.0 either and which our port already documents as unsupported, plus a
  class rename and a `toString` escaping change that does not affect parsing.
- **`ForUtil`/`DirectWriter`** — no constant changed. `BLOCK_SIZE == 256` in
  both; `expand8` was only delegated to `VectorUtil`.
- **`Lucene90DocValuesProducer`** — the whole 227-line addition is query-time
  bulk/SIMD range evaluation. No entry parsing, no encoding, no `DirectReader`
  usage changed.

**Batches whose conclusions need revisiting:**

- **b10-merge** — finding 1. Its "ported line by line and verified against real
  Lucene" claim was true of the fixture but the fixture does not reach the
  `maxMergeAtOnce` bound. Now fixed and covered by a new unit test.
- **b5-postings / whichever batch touched `postings_writer.rs`'s block-encoding
  choice** — finding 2. The port was *moved onto* `main` by a sweep batch, with
  a test written to lock it there. Now reverted.
- **b12-search-core / c11-occur-filter / c12-search-features-2** (whichever owns
  the `BooleanQuery::rewrite` scope note) — finding 3. The "impossible without a
  `Similarity`" argument is void for 10.5.0.
- **c9-check-index** — finding 4. `testHnswGraphs` has no 10.5.0 counterpart.
- **b1-util-store** — finding 5, documentation only.

Everything else in the sweep stands: for the other 26 classes the tag and `main`
say the same thing about what we ported.

**Where a fixture already pins the behaviour against 10.5.0 jars, the
source-reading error was harmless.** That covers `SegmentInfos` (`GenSegmentInfos`
+ `VerifySegmentInfos` — and it is the one class where `main` bumps the on-disk
format version, so the fixture is exactly what kept us on `VERSION_86`), stored
fields, term vectors, doc values, norms, field infos, points, live docs, FST,
block tree, and the postings *read* path. The two real correctness findings both
landed in paths with **no fixture coverage of the specific decision**: a merge
grouping the fixture's scenarios never reach, and a writer encoding choice both
readers accept.

## Gate

`cargo fmt --all` clean. `cargo clippy --workspace --all-targets -- -D warnings`
clean. `cargo test --workspace`: 78 suites, 0 failures. The
`merge_policy_fixtures` differential test (generated by `GenMergePolicy.java`
against the real 10.5.0 jars) still passes after finding 1's fix, which is the
evidence that restoring `maxMergeAtOnce` did not disturb any grouping the
fixture does cover.

Clippy and the index crate were transiently blocked twice by
`crates/lucene-index/src/index_writer.rs` (owned by `c17-index-sort`,
mid-edit); retried until clean. No file owned by `c15`/`c16`/`c17` was
modified by this batch.
