# b10-merge — M2 sweep

Files swept:

- `crates/lucene-index/src/merge.rs` (9 940 lines; ~2 800 code, ~7 100 test)
- `crates/lucene-index/src/merge_policy.rs`

Java source of truth: `/home/tuong/work/lucene` @ `091a987a14d` (Lucene 10.5.0).

Gate: `cargo fmt -p lucene-index -p lucene-ffi --check` clean,
`cargo clippy -p lucene-index --all-targets -- -D warnings` clean, and
`cargo test -p lucene-index -p lucene-ffi` green: 393 `lucene-index` lib
tests (up from 344) plus 16 integration tests including the new
`merge_policy_fixtures` differential suite. `cargo llvm-cov -p lucene-index
--summary-only`: `merge.rs` 98.56% lines, `merge_policy.rs` 98.63% lines, both
above the 95% bar. (`cargo test --workspace` currently has one unrelated
failure and one unformatted file in `crates/lucene-search`, batch b12/b13's
in-flight work.)

A Tier 2 `quality-reviewer` pass was run on this batch's diff against the Java
source; findings 32-38 below are its output and are all resolved.

---

## `crates/lucene-index/src/merge_policy.rs`

Java counterparts:

- `lucene/core/src/java/org/apache/lucene/index/TieredMergePolicy.java`
- `lucene/core/src/java/org/apache/lucene/index/MergePolicy.java` (`size`,
  `findFullFlushMerges`, `isMerged`, `MergeSpecification`, `OneMerge`)
- `lucene/core/src/java/org/apache/lucene/index/{LogMergePolicy,LogByteSizeMergePolicy,NoMergePolicy,MergeScheduler,ConcurrentMergeScheduler,MergeTrigger,MergeRateLimiter}.java`
  — **no Rust counterpart, by design** (see findings 12–13).

### Method correspondence (before the sweep)

| Java | Rust (before) | Verdict |
|---|---|---|
| `TieredMergePolicy.findMerges` | `find_merges` | **divergent** — different algorithm entirely |
| `TieredMergePolicy.doFindMerges` | — | **missing** |
| `TieredMergePolicy.score` | `effective_score` | **divergent** — invented formula |
| `TieredMergePolicy.getSortedBySegmentSize` | inline `sort_by(effective_score)` | **divergent** — ascending by invented score vs descending by size |
| `TieredMergePolicy.floorSize` | inline `max()` in `effective_score` | identical |
| `TieredMergePolicy.getMaxAllowedDocs` | — | **missing** |
| `TieredMergePolicy.findForcedMerges` | `find_forced_merges` | **divergent** — `segments[..excess]` in caller order |
| `TieredMergePolicy.findForcedDeletesMerges` | `find_forced_delete_merges` | **divergent** — one unbounded group |
| `MergePolicy.size` (pro-rated bytes) | — | **missing** |
| `MergePolicy.findFullFlushMerges` | — | missing, intentional (finding 13) |
| `MergePolicy.isMerged` | — | missing, intentional (finding 13) |
| `SegmentSizeAndDocs` | `SegmentStat` | close; missing the pro-rated/raw split |
| `mergeContext.getMergingSegments()` | — | **missing** |
| `setDeletesPctAllowed` / `deletesPctAllowed` | — | **missing** |
| `setTargetSearchConcurrency` | — | **missing** |
| `setSegmentsPerTier` (double, `>= 2`) | `segments_per_tier: usize` | divergent default, no validation |
| `setMaxMergedSegmentMB` | `max_merged_segment_size` | divergent default |
| `setFloorSegmentMB` | `floor_segment_size` | identical default |
| `setForceMergeDeletesPctAllowed` | `force_merge_deletes_pct_allowed` | identical default |
| — | `reclaim_weight` | not-in-Java (port invention) |
| — | `max_merge_at_once` | not-in-Java in 10.x (`mergeFactor = (int) segsPerTier`) |
| — | `segment_byte_size` | not-in-Java (glue for `SegmentCommitInfo.sizeInBytes()`) |

### Findings

1. **[CORRECTNESS] `segmentsPerTier` default was 10, Java's is 8.**
   `TieredMergePolicy.java:79` — `private double segsPerTier = 8.0;`. The port
   shipped `segments_per_tier: 10` (Lucene 7's value). It is the single most
   load-bearing constant in the policy: it is the per-level segment budget, it
   *is* `mergeFactor` (`findMerges`: `final int mergeFactor = (int) segsPerTier;`),
   and it is the `hitTooLarge` skew denominator in `score()`. A 25%-too-high
   value systematically under-merges.
   **Fixed** — `MergePolicyConfig::default().segments_per_tier == 8`, asserted by
   `default_config_matches_real_lucene_defaults`.

2. **[CORRECTNESS] `maxMergedSegmentBytes` default was 5 000 MiB, Java's is 5 GiB.**
   `TieredMergePolicy.java:76` — `5 * 1024 * 1024 * 1024L` = 5 368 709 120.
   The port had `5_000 * 1024 * 1024` = 5 242 880 000, a 2.4% shortfall — the
   value you would get from `setMaxMergedSegmentMB(5000)`, not from the field
   initialiser. **Fixed**, asserted by the same test.

3. **[MISSING] `deletesPctAllowed` (default 20.0) was absent entirely.**
   It drives three separate behaviours in `findMerges`/`doFindMerges`:
   (a) `allowedDelCount = (int)(deletesPctAllowed * totalMaxDoc / 100)`, the
   delete budget that makes the policy merge *even when the segment count is
   already within budget*; (b) the over-half-`maxMergedSegmentBytes` exclusion
   is conditional on `totalDelPct <= deletesPctAllowed || segDelPct <= deletesPctAllowed`,
   so a delete-heavy huge segment stays mergeable; (c) it is the escape hatch on
   the 1.5x growth guard (`maxCandidateSegmentSize.delCount < maxDoc * deletesPctAllowed / 100`).
   Without it the port would never merge purely to reclaim deletes.
   **Fixed** — `MergePolicyConfig::deletes_pct_allowed`, all three uses ported;
   tests `deletes_alone_can_trigger_a_merge_within_the_segment_count_budget`,
   `over_half_max_size_segment_with_heavy_deletes_is_still_merged`,
   `growth_guard_is_bypassed_when_the_biggest_input_is_delete_heavy`.

4. **[MISSING] `targetSearchConcurrency` (default 1) was absent entirely.**
   Java reserves a whole segment slot for each of the first
   `targetSearchConcurrency - 1` segments, floors `allowedSegCount` at
   `targetSearchConcurrency - tooBigCount`, and caps every merge's live-doc
   count at `getMaxAllowedDocs = ceilDiv(totalMaxDoc - totalDelDocs, targetSearchConcurrency)`
   so search work stays splittable into that many similar slices.
   **Fixed** — config knob plus `max_allowed_docs`; tests
   `target_search_concurrency_bounds_merged_doc_count`,
   `max_allowed_docs_is_ceiling_division`.

5. **[CORRECTNESS] The `MergeScore` formula was invented, not ported.**
   Java (`TieredMergePolicy.score`, lines ~666–740):
   ```
   skew       = hitTooLarge ? 1.0/(int)segsPerTier
                            : floorSize(biggest) / Σ floorSize(seg)
   mergeScore = skew · (Σ proratedBytes)^0.05 · (Σ proratedBytes / Σ rawBytes)^2
   ```
   The port scored `floorSize(size) * (1 - clamp(reclaim_weight * delRatio))`
   — no skew term at all, so it had no defence against the lopsided merges the
   skew factor exists to prevent, and its "reclaim" term is a linear discount
   rather than Java's quadratic `nonDelRatio` penalty on raw-vs-live bytes.
   **Fixed** — `merge_score` is the Java formula line for line; `reclaim_weight`
   is redefined as the exponent Java hardcodes as `2` (default `2.0`, `0.0`
   disables reclaim weighting). Tests: `merge_score_matches_the_java_formula`
   (both `hitTooLarge` branches, exact float equality to 1e-12),
   `merge_score_favours_reclaiming_deletes`, `merge_score_favours_lower_skew`,
   `floor_segment_size_flattens_skew_among_tiny_segments`.

6. **[CORRECTNESS] Segment size was never pro-rated by deletes.**
   `MergePolicy.size` (line 720) returns `byteSize * (1 - delCount/maxDoc)`,
   and *every* size comparison in `TieredMergePolicy` uses that; only
   `score()`'s `totBeforeMergeBytes` reads the raw `SegmentCommitInfo.sizeInBytes()`.
   The port fed raw bytes everywhere, so a 90%-deleted 1 GB segment looked like
   1 GB of merge work instead of 100 MB, and `nonDelRatio` could not exist at all.
   **Fixed** — `prorated_size` is public and documented; `SegmentStat::size_bytes`
   is now explicitly the raw figure. Test `prorated_size_matches_merge_policy_size`
   (including the `maxDoc <= 0` no-proration branch).

7. **[CORRECTNESS] `getSortedBySegmentSize` order was wrong.**
   Java sorts **descending by pro-rated size**, name ascending on ties, and the
   whole of `doFindMerges` depends on it: `candidate.get(0)` is assumed to be
   the largest input (the 1.5x guard and the skew numerator both read it), and
   `findForcedMerges` walks the list *backwards* to bin-pack smallest-first.
   The port sorted **ascending by its invented score**. **Fixed** —
   `sorted_by_segment_size`; test `sorted_by_segment_size_is_size_descending_then_name`
   (with a pro-rated tie to prove the name tiebreak).

8. **[MISSING] The `allowedSegCount` budget walk was absent.**
   Java computes the allowed segment count by walking size levels upward from
   `max(minSegmentBytes, floorSegmentBytes)`, multiplying by `mergeFactor` each
   level and accumulating `segsPerTier` per level until the remainder fits, then
   flooring at `segsPerTier` and at `targetSearchConcurrency - tooBigCount`.
   The port used the flat rule "merge iff count > segments_per_tier".
   **Fixed** — ported verbatim, with one deliberate hardening: `level_size` is
   clamped to `>= 1` and `segs_per_tier` to `>= 2.0`, because Java's own
   preconditions (real files are never 0 bytes; the setter rejects
   `segsPerTier < 2.0`) are not enforceable on this port's unit-agnostic
   `size_bytes`, and violating either makes the loop non-terminating. Tests
   `degenerate_zero_size_segments_terminate`, `segments_per_tier_below_two_is_clamped`.

9. **[MISSING] Six `doFindMerges` behaviours had no counterpart.**
   All now ported and individually tested:
   - the multi-cycle `toBeMerged` loop that emits several disjoint merges per
     call (the port emitted greedy fixed-size chunks off one sorted list);
   - the start-at-every-index candidate search with "packing" (`continue` past
     a segment that would overflow, to try a smaller one);
   - the "keep packing past `mergeFactor` while still under `floorSegmentBytes`"
     escape hatch (`merge_factor_cap_respected_unless_below_floor`);
   - the `maxMergedSegmentBytes` / `allowedDocCount` caps inside candidate
     building (`max_merged_segment_size_caps_a_merge`);
   - the "result must be ≥50% larger than the biggest input" O(N²) guard and
     its delete-heavy bypass
     (`pathological_growth_guard_skips_a_merge_that_barely_grows_the_biggest_input`);
   - `haveOneLargeMerge` (at most one `hitTooLarge` merge per call) and
     `maxMergeIsRunning`.

10. **[MISSING] `mergeContext.getMergingSegments()` exclusion was absent.**
    Java removes already-merging segments from the eligible list, accumulates
    their bytes into `mergingBytes` (which gates `maxMergeIsRunning`), and counts
    only their *live* docs towards `totalMaxDoc` because their deletes are
    already being reclaimed. **Fixed** — exposed as an explicit argument rather
    than ambient state: `find_merges_excluding(segments, merging, config)` and
    `find_forced_delete_merges_excluding`; `find_merges`/`find_forced_delete_merges`
    delegate with an empty set. Tests
    `merging_segments_are_excluded_from_the_eligible_set`,
    `find_forced_delete_merges_excludes_merging_segments`.

11. **[CORRECTNESS] `findForcedMerges` merged the wrong segments, in the wrong order.**
    The port returned `segments[..len - max + 1]` — the first N in *caller* order,
    never sorted, never size-aware, with no `maxMergeBytes` derivation and no
    bail-outs. Java: derives `maxMergeBytes` (`Long.MAX_VALUE` for
    `maxSegmentCount == 1`, else `max(totalMergeBytes/maxSegmentCount, maxMergedSegmentBytes) * 1.25`),
    drops delete-free segments already at/above it, bails out when nothing has
    deletes and the count is already within target, special-cases "everything
    fits in one segment", and otherwise **bin-packs from the smallest end
    upward** so the biggest segments are left alone.
    **Fixed** — signature is now `find_forced_merges(segments, max_segment_count, config)`
    (no external callers existed) with `UNLIMITED_SEGMENT_COUNT` for Java's
    `Integer.MAX_VALUE`. Tests: `..._down_to_one_segment`,
    `..._no_op_when_already_at_target`, `..._single_clean_segment_to_one_is_a_no_op`,
    `..._rewrites_a_lone_deleted_segment`, `..._bin_packs_from_the_smallest_end`,
    `..._drops_oversized_clean_segments`, `..._empty_input`.

12. **[CORRECTNESS] `findForcedDeletesMerges` could produce arbitrarily large segments.**
    `TieredMergePolicy`'s class javadoc states outright: *"findForcedDeletesMerges
    should never produce segments greater than maxSegmentSize."* Java achieves
    that by delegating to the same `doFindMerges` engine with
    `mergeFactor = Integer.MAX_VALUE`, `allowedSegCount = Integer.MAX_VALUE`,
    `allowedDelCount = 0` and the real `allowedDocCount`. The port returned a
    single group containing *every* over-threshold segment, so
    force-merge-deletes on a large index proposed one unbounded merge.
    **Fixed** — same delegation; test
    `find_forced_delete_merges_respects_max_merged_segment_size` (10 segments
    pro-rating to 400 bytes each under a 1 000-byte cap must come back as
    disjoint groups of ≤2, covering every qualifying segment exactly once).

13. **[INTENTIONAL] No `MergeScheduler`, `MergeRateLimiter`, `OneMerge`,
    `MergeTrigger`, `findFullFlushMerges`, `LogMergePolicy`, `NoMergePolicy`.**
    Recorded, not fixed. `merge_policy.rs` is a pure decision function; there is
    no background merging in this port, so `MergeScheduler`/`ConcurrentMergeScheduler`
    (thread pool, `maxMergeCount`/`maxThreadCount` auto-tuning, IO throttling via
    `MergeRateLimiter`) and `MergeTrigger` have nothing to schedule.
    `findFullFlushMerges` is `findMerges` filtered to merges whose every input is
    below `maxFullFlushMergeSize()` (= `floorSegmentBytes` here) and only matters
    when merges run concurrently with a commit. `LogByteSizeMergePolicy`/
    `NoMergePolicy` are alternative policies, not gaps in this one. All now stated
    explicitly in the module doc rather than being silently absent.

14. **[INTENTIONAL] `segmentsPerTier` is `usize`, not `double`.**
    Java accepts fractional values (`8.5`). Changing the field's type would break
    the FFI ABI and in-flight callers for no realistic gain; the value is clamped
    to `>= 2` exactly where Java's setter validates. Documented in the module doc.

15. **[INTENTIONAL] `max_merge_at_once` is retained as an explicit knob.**
    `maxMergeAtOnce` was removed from `TieredMergePolicy` in Lucene 9; 10.5 uses
    `mergeFactor = (int) segsPerTier`. The port keeps the field (it is part of the
    `lucene-ffi` ABI and `lucene-search`'s test configs) but its default now equals
    the default `segments_per_tier`, so an untouched config behaves exactly like
    Java. Note `score()`'s `hitTooLarge` skew still uses `segs_per_tier`, not this
    knob — faithful to Java, which reads `(int) segsPerTier` there directly.

16. **[PERF] Candidate search is O(N²) per cycle, same as Java.** `doFindMerges`
    tries a candidate starting at every index; each candidate scans forward until
    a cap is hit. Java is identical, including the `break` when a short candidate
    is found after a best already exists. No divergence; recorded so a future
    reader does not "fix" it.

17. **[PERF] `remaining.retain(...)` per cycle is O(N) against Java's `Iterator.remove()`
    on an `ArrayList`, which is O(N²) for N removals.** The Rust side is
    strictly better here (one compacting pass vs repeated `System.arraycopy`),
    and cycles are bounded by the number of emitted merges. No action.

### Verdict

**Swept clean, and now differentially verified against real Lucene.**
`merge_policy.rs` is a faithful port of `TieredMergePolicy`'s three decision
entry points, its scoring, and its budget model, with every remaining gap named
in the module doc. 40 unit tests written against the Java source, plus the new
33-scenario Java fixture (finding 33) which every scenario passes on the nose —
same groups, same group contents, same order, first run. Two behavioural
consequences were absorbed by callers: `index_writer.rs`'s `tight_merge_policy()` test helper and
`tests/merge_policy_to_merge_integration.rs` both needed a `floor_segment_size`
above their (tiny, synthetic) segments, because with a zero floor the *real*
budget walk correctly rules three tiny segments in-budget where the old flat
`count > segments_per_tier` rule did not. Both changes are one line plus a
comment explaining why; `lucene-ffi`'s `ffi_writer_set_merge_policy` gained
`..MergePolicyConfig::default()` so the two new knobs take their Lucene defaults.

---

## `crates/lucene-index/src/merge.rs`

Java counterparts:

- `index/SegmentMerger.java`, `index/MergeState.java`, `index/DocIDMerger.java`,
  `index/MappingMultiPostingsEnum.java`, `index/MultiPostingsEnum.java`,
  `index/MultiTerms.java`, `index/ReaderSlice.java`
- `index/FieldInfos.java` (`Builder.add`, `FieldNumbers.addOrGet`),
  `index/FieldInfo.java` (`verifySameSchema` and its `verifySame*` helpers)
- `codecs/{StoredFieldsWriter,TermVectorsWriter,NormsConsumer,DocValuesConsumer,PointsWriter,FieldsConsumer,KnnVectorsWriter}.java`
- `codecs/lucene90/compressing/Lucene90CompressingStoredFieldsWriter.java`
  (`merge`, `copyChunks`, `copyOneDoc`, `getMergeStrategy`, `tooDirty`) and
  `codecs/MatchingReaders.java`

### Method correspondence

| Java | Rust | Verdict |
|---|---|---|
| `SegmentMerger.merge` | `merge_stored_only_segments` | divergent (materialising, not streaming; see 24–25) |
| `SegmentMerger.merge` + `MultiSorter`/index sort | `merge_sorted_stored_only_segments` | equivalent; see 23 |
| `SegmentMerger.mergeFieldInfos` / `FieldInfos.Builder.add` | `reconcile_field_numbers` | **was divergent — fixed (18)** |
| `FieldInfo.verifySameSchema` | — | **was missing — fixed (18)** |
| `SegmentMerger.shouldMerge` | — | missing (22) |
| `StoredFieldsWriter.merge` (+ `MergeVisitor`) | inline loop in both entry points | equivalent, minus the field-number-alignment fast path (26) |
| `Lucene90CompressingStoredFieldsWriter.merge` BULK/DOC strategies | — | **missing — PERF (24)** |
| `MatchingReaders` | — | missing (24) |
| `TermVectorsWriter.merge` | `merge_term_vectors` | **was divergent — fixed (19)** |
| `NormsConsumer.mergeNormsField` | `merge_norms` | divergent, scoped (21) |
| `DocValuesConsumer.mergeNumericField` | `merge_numeric_doc_values` | divergent, scoped (21) |
| `DocValuesConsumer.mergeBinaryField` | `merge_binary_doc_values` | divergent, scoped (21) |
| `DocValuesConsumer.mergeSortedField` (+ `OrdinalMap`) | `merge_sorted_doc_values` | equivalent result, different mechanism (28) |
| `DocValuesConsumer.mergeSortedNumericField` | `merge_sorted_numeric_doc_values` | equivalent |
| `DocValuesConsumer.mergeSortedSetField` (+ `OrdinalMap`) | `merge_sorted_set_doc_values` | equivalent result, different mechanism (28) |
| `FieldsConsumer.merge` / `MappedMultiFields` / `MappingMultiPostingsEnum` | `merge_postings` | equivalent result, different mechanism — PERF (27) |
| `PointsWriter.merge` | `merge_points` | **was divergent — fixed (19)**; PERF (29) |
| `KnnVectorsWriter.merge` | — | missing, out of scope (30) |
| `MergeState.DocMap` | `build_doc_id_maps` + `mapped_doc_id` | **was a `HashMap` — fixed (25)** |
| `DocIDMerger.SequentialDocIDMerger` | `concat_doc_order` | equivalent |
| `DocIDMerger.SortedDocIDMerger` (priority queue) | k-way linear scan in `merge_sorted_stored_only_segments` | equivalent result, O(D·S) vs O(D·log S) — PERF (23) |
| `SegmentInfo.minVersion` propagation | hardcoded to the caller's version | divergent, unreachable today (20) |
| `SegmentInfo.hasBlocks` propagation | hardcoded `false` | divergent, unreachable today (20) |

### Findings

18. **[CORRECTNESS] `FieldInfos.Builder.add`'s schema reconciliation was not ported.**
    Java runs `verifySameSchema` for every `FieldInfo` of every merged segment and
    throws on any disagreement in `indexOptions`, `omitNorms`/`storeTermVector`
    (when the field is indexed), `docValuesType`, `docValuesSkipIndexType`, the
    points dimension triple, or the vector options — then ORs `hasPayloads` in.
    The port took the **first-seen source's `FieldInfo` verbatim** and never
    compared, with only two ad-hoc after-the-fact checks inside `merge_postings`
    that fired only when postings data happened to be supplied.
    Consequences, all reachable:
    - a source with `store_term_vectors = true` merged against a first-seen
      source without it produced a `.fnm` claiming no term vectors while
      `merge_term_vectors` wrote them — a segment whose metadata and data
      disagree;
    - same for `omit_norms` (a `.fnm` promising norms the merge did not write, or
      the reverse) and `doc_values_type`;
    - `index_options` and points dims were unchecked whenever the corresponding
      data was not supplied through `MergeSource`.
    **Fixed** — `reconcile_field_numbers` now returns `Result` and ports
    `FieldInfos.Builder.add` whole: `verify_same_schema` (a direct port of
    `FieldInfo.verifySameSchema`, checks in Java's order) plus the
    `setStorePayloads` OR. New `Error::FieldSchemaDisagreement` carries the
    attribute name; `index_options` reuses the existing
    `PostingsIndexOptionsDisagreement`, points dims reuse
    `PointsShapeDisagreement`/`PointsIndexDimsDisagreement`. The now-redundant
    per-source checks inside `merge_postings` were removed. Six new tests, one
    per attribute group, including the "not compared when the field is not
    indexed" half of Java's guard.

    Sub-finding, same fix: **`store_payloads` disagreement was a hard error**
    where Java merges it. `Error::PostingsPayloadsDisagreement` is gone; the
    merged field now stores payloads if *any* source does, and a payload-free
    source's occurrences come through as empty `Position::payload`s, which the
    postings writer already documents as "no payload at this occurrence". The
    old rejection test is now
    `a_source_with_payloads_ors_into_the_merged_fields_store_payloads`, which
    round-trips the merged segment and asserts `.fnm` says `store_payloads`,
    source 0's term has an empty payload and source 1's keeps `b"pay"`.

19. **[MISSING] The blanket "sparse across sources is a hard error" rule rejected
    merges Lucene performs routinely.**
    Adding a field (or turning term vectors on) for an index that already has
    segments is normal; those older segments have no `FieldInfo` for the field and
    simply contribute nothing. Java:
    - `FieldsConsumer.merge` → `MultiFields` skips a reader whose `terms(field)`
      is null;
    - `PointsWriter.merge` → `if (readerFieldInfo == null) continue;` and
      `if (getPointDimensionCount() == 0) continue;`;
    - `TermVectorsWriter.merge` → `mergeState.termVectorsReaders[i]` may be null →
      `vectors = null` → `addAllDocVectors(null, ...)` writes a vector-less doc.

    The port raised `PostingsFieldMissingInSource`, `PointsFieldMissingInSource`
    and `TermVectorsReaderMissingInSource`. This is reachable straight through
    `IndexWriter`: commit, then `set_postings_field(Some(..))` (or
    `set_term_vector_field`), commit again, and the automatic merge fails.
    **Fixed**, keeping the useful half of the check: a source whose own
    `FieldInfos` **never saw the field** now contributes nothing; a source that
    **declares** the field but whose `MergeSource` supplied no data for it is
    still a hard error, because that is a caller wiring bug rather than index
    evolution. (Declaring the field with *different* `index_options` is caught
    earlier by finding 18, exactly as in Java.)
    `Error::TermVectorsReaderMissingInSource` is removed;
    `PostingsFieldMissingInSource`/`PointsFieldMissingInSource` were re-documented
    to the narrower meaning. Tests:
    `a_source_that_never_saw_the_postings_field_simply_contributes_no_terms`,
    `a_source_that_never_saw_the_points_field_simply_contributes_no_points`, and
    the rewritten `term_vectors_merge_across_two_sources_with_deletions_and_a_source_with_none`
    (source 1's doc must come back as an empty term-vectors document, source 0's
    surviving doc must keep its vectors).

20. **[MISSING] `minVersion` and `hasBlocks` are not propagated from the sources.**
    `SegmentMerger`'s constructor computes
    `minVersion = min over readers of reader.getMetaData().minVersion()` (null if
    any reader's is null) and asserts the caller left it unset; `IndexWriter`
    propagates `hasBlocks`. Both entry points here hardcode
    `min_version: Some(lucene_version)` and `has_blocks: false`.
    **Recorded, not fixed.** Unreachable today: this port only ever merges
    segments it wrote itself, always with the caller-supplied version, and it
    never writes doc blocks — so the two values are correct in every reachable
    case. Fixing it properly means adding per-source `min_version`/`has_blocks`
    to `MergeSource`, which is an exhaustive-struct-literal break across ~85
    call sites *and* `index_writer.rs`, currently owned by batch b9. Carry-over.

21. **[INTENTIONAL] Doc-values/norms keep the strict "every live-doc-contributing
    source must supply a dense value" rule.**
    Java's `DocValuesConsumer`/`NormsConsumer` merges skip a reader that lacks the
    field and produce a *sparse* merged field. This port's writers
    (`write_single_dense_*`, `norms::write_single_dense_field`) are dense-only, so
    a sparse merged field is not expressible; erroring beats silently dropping
    values. Unchanged and still documented in the module doc. Blocked on
    `IndexedDISI` write support in `lucene-codecs` (b6's territory).

22. **[MISSING] `SegmentMerger.shouldMerge()` — no zero-doc guard.**
    Java refuses to merge when `segmentInfo.maxDoc() == 0` and `IndexWriter` drops
    the resulting segment entirely. Merging N fully-deleted sources here writes a
    real 0-doc segment. **Recorded, not fixed**: `merge_stored_only_segments` is a
    low-level primitive and producing what it was asked for is defensible; the
    drop belongs in `IndexWriter::apply_merge` (b9's file). Not newly introduced
    by this batch — the old policy also scored fully-deleted segments best.
    Carry-over for b9/b11.

23. **[PERF] The sorted merge is a linear scan over sources per document.**
    `merge_sorted_stored_only_segments` picks the smallest head by scanning all S
    sources each step: O(D·S). Java's `DocIDMerger.SortedDocIDMerger` keeps a
    `PriorityQueue` of size `maxCount - 1` plus a `queueMinDocID` fast path that
    avoids touching the heap entirely while the current sub stays ahead — O(D)
    amortised in the common case, O(D·log S) worst case. **Recorded, not fixed**:
    the crossover is around S ≈ 8–16 sources, `max_merge_at_once` defaults to 8,
    and the existing in-code comment already justifies the choice. Worth revisiting
    only if forced merges of many segments become a real workload. Note the tie
    break is equivalent: `compare_heads` falls back to source index then doc id,
    which is what `MultiSorter` does.

24. **[PERF] No bulk-copy stored-fields merge path — the largest cost divergence
    in this file.**
    `Lucene90CompressingStoredFieldsWriter.merge` picks one of three strategies
    per reader via `getMergeStrategy` + `MatchingReaders`:
    - **BULK** (`copyChunks`) when the reader is the same codec/version, same
      `compressionMode`, same `chunkSize`, has **no live-docs bitset**, is not
      `tooDirty`, and its field numbers align: the already-compressed chunk bytes
      are `copyBytes`'d straight through, rewriting only the per-chunk `docBase`
      varint and the index entry. **Zero decompression, zero recompression, zero
      per-field work.**
    - **DOC** (`copyOneDoc`) when the codec matches but deletions or dirtiness
      rule out chunk copying: the doc's *serialised* bytes are copied without
      being parsed into fields.
    - **VISITOR** only as the fallback.

    This port always takes the VISITOR-equivalent path, and a heavier one:
    `StoredFieldsReader::document(doc_id)` materialises an owned `Document`
    (a `Vec<StoredField>` with owned `String`/`Vec<u8>` per field), every merged
    doc is accumulated into `merged_docs: Vec<Document>`, and
    `write_best_speed` re-compresses everything.

    Per document that means: one LZ4 block decompress + one allocation per stored
    field + one `HashMap` lookup per field (26) + one LZ4 recompress, versus Java's
    ~one `memcpy` per 16 KB chunk in the BULK case. For a deletion-free merge of
    same-codec segments — which is the overwhelmingly common case, and exactly
    what `IndexWriter::auto_merge` produces — this is an order-of-magnitude gap,
    and it is CPU that scales with total stored-field bytes, not doc count.

    **Recorded, not fixed.** Not fixable inside this batch: it needs a new
    `lucene-codecs` stored-fields API (a `MatchingReaders` equivalent, raw chunk
    access on the reader, and a writer that accepts pre-compressed chunks +
    serialised docs), and `lucene-codecs` is under concurrent edit. This extends
    the existing b3 carry-over ("stored-fields writer API takes `&[Document]`
    rather than streaming") with the concrete merge-side consequence. No
    microbenchmark: there is no second implementation to compare against, and the
    analytic difference (memcpy vs decompress+parse+allocate+recompress) is not
    marginal.

25. **[PERF, fixed] `MergeState.DocMap` was a `HashMap<i32, i32>` in the innermost
    merge loop.**
    Java's `DocMap` is a dense array-backed old→new lookup returning `-1` for a
    deleted doc, precisely because it is hit once per posting, per point, per
    doc-values entry. `build_doc_id_maps` built a `HashMap` per source and
    `merge_postings`/`merge_points` did a hash probe per posting.
    **Fixed** — `build_doc_id_maps` now returns `Vec<Vec<i32>>` (index = source doc
    id, `-1` = deleted, sized to the source's highest live doc id + 1) with an
    inlined `mapped_doc_id` accessor. Each lookup becomes a bounds check + index +
    sign test instead of hashing an `i32` and chasing a bucket, and the per-entry
    hash-table overhead disappears from the allocation. Test
    `doc_id_maps_are_dense_arrays_with_minus_one_for_deleted_docs` pins the
    representation and every edge (deleted doc, past-the-end, negative, and a
    fully-deleted source's empty map); `concat_doc_order_walks_sources_in_order`
    pins the sequential merger's order.

26. **[PERF, partially fixed] Field-number remapping does a hash lookup per field
    per doc, and a linear scan per (field, source).**
    Java's `StoredFieldsWriter.MergeVisitor` checks once per reader whether the
    field numbers already align and, if so, sets `remapper = null` so `remap()` is
    a no-op for every field of every doc.
    Two separate costs here:
    (a) the per-field `field_number_map.get(&field.field_number)` in both entry
    points' stored-field loops. **Recorded, not fixed** — the identity-map fast
    path is worth having but is dwarfed by finding 24 in the same loop, and both
    want the same restructuring.
    (b) resolving "what does this source call merged field N?" was
    `map.iter().find(|(_, &merged)| merged == n)` — a linear scan of the source's
    whole forward map, run once per (candidate field, source), i.e.
    O(fields² · sources) on a wide schema. **Fixed** —
    `invert_field_number_maps` builds the reverse maps once per merge and
    `merge_postings`/`merge_points` do a single hash lookup. (The doc-values/norms
    helpers still scan, but they are limited to one candidate field per call by
    construction, so the scan is O(fields) total there.)

27. **[PERF] The postings merge materialises everything and re-seeks per term.**
    Java streams: `FieldsConsumer.merge` wraps the readers in `MultiFields` +
    `MappedMultiFields`, so the writer pulls a single k-way-merged `TermsEnum`
    forward once, and each term's `MappingMultiPostingsEnum` walks each sub's
    already-positioned `PostingsEnum` exactly once, remapping doc ids as it goes.
    Nothing is materialised; each source's dictionary is traversed once, in order.

    `merge_postings` instead:
    - builds `all_terms: BTreeSet<Vec<u8>>` — one heap allocation and one tree
      insert per (term, source), holding every distinct term of the merged field
      in memory at once;
    - for each term, calls `field_terms.postings(&term, ...)` per source — a fresh
      blocktree/FST seek from the root, not a cursor step;
    - **and**, when positions are indexed, calls `field_terms.positions(&term, ...)`
      per source as well, which re-seeks *and* re-decodes that term's docs and
      freqs a second time (the existing in-code comment acknowledges this);
    - accumulates the whole merged field into `Vec<TermPostings>` (per term: a
      `Vec<(i32,i32)>` plus a `Vec<Vec<i32>>`, `Vec<Vec<(i32,i32)>>` and
      `Vec<Vec<Vec<u8>>>`) before `write_fields` sees any of it;
    - computes `docCount` with a `HashSet<i32>` over every posting, where Java
      tracks it incrementally as it writes.

    So: 2 dictionary seeks and 2 docs/freqs decodes per (term, source) against
    Java's 1 cursor step and 1 decode, plus O(all postings) resident memory
    against O(one term's postings). **Recorded, not fixed** — the fix is a
    streaming merged-terms cursor plus a combined postings+positions read API in
    `lucene-codecs` (which is also where the b4 carry-over A1, "blocktree
    materialises the whole term dictionary at open", lives — these two want the
    same lazy `SegmentTermsEnum`). Out of this batch's files.

28. **[INTENTIONAL] No `OrdinalMap` for SORTED/SORTED_SET doc values.**
    Java builds an `OrdinalMap` (a compressed global-ordinal → per-segment-ordinal
    mapping) so merged ordinals can be remapped without re-materialising terms.
    This port resolves each doc's ordinal to *term bytes* through that source's own
    dictionary and lets `write_single_dense_sorted*_field` rebuild and dedupe the
    merged dictionary by bytes. Same output, no remapping table to get wrong;
    costs one dictionary decode per source (already done for the write anyway) and
    a `Vec<u8>` clone per value. Documented; keep.

29. **[PERF] Points merge is a full re-index, as is Java's *default* — but Java's
    real codec overrides it.** `PointsWriter.merge`'s default implementation is a
    streaming visitor that re-indexes every live point, which is what
    `merge_points` does. `Lucene90PointsWriter` overrides `merge` to call
    `BKDWriter.merge`, a one-pass merge over the already-sorted per-segment
    trees, skipping the global sort entirely. `merge_points` additionally holds
    every live point (`Vec<(i32, Vec<u8>)>`, one allocation per packed value)
    before `points::write` sorts and builds the tree. **Recorded, not fixed** —
    needs a `BKDWriter.merge` equivalent in `lucene-codecs::points`.

30. **[MISSING] `KnnVectorsWriter.merge` has no counterpart.**
    Out of scope: the port has no HNSW writer at all (already a b7 carry-over).
    `MergeSource` carries no vector data, and `merge_stored_only_segments` writes
    no vector files, so there is nothing to silently drop — a vector-bearing
    segment simply cannot be constructed by this port yet. Recorded for
    completeness.

31. **[INTENTIONAL] No `checkAborted`/`checkIntegrity`/`OneMerge` plumbing.**
    Java calls `mergeState.checkAborted()` and `reader.checkIntegrity(oneMerge)`
    per reader in every codec merge entry point. This port has no abortable merge
    and validates checksums at `open()` time instead of at merge time
    (`checksum_verify.rs`, b11). Recorded.

### Verdict

**Swept, with three PERF items left open** (24 stored-fields bulk copy,
27 postings streaming, 29 BKD merge) — all three are `lucene-codecs` API
changes, all three are recorded as carry-overs, and all three are blocked on
crates outside this batch that are under concurrent edit. Everything classified
CORRECTNESS or MISSING and reachable is fixed:

- cross-source `FieldInfo` schema verification + `store_payloads` OR (18);
- Java-tolerant handling of a source that never saw a field, for postings,
  points and term vectors (19);
- the dense `MergeState.DocMap` representation (25) and the reverse
  field-number maps (26b) as contained PERF fixes.

Two MISSING items are recorded as unreachable-today with a named blocker
(20 `minVersion`/`hasBlocks`, 22 zero-doc merge).

---

## Tier 2 review findings (`quality-reviewer`, run on this batch's diff)

32. **[CORRECTNESS] The `store_payloads` OR dropped `setStorePayloads`' own
    guard.** Finding 18's sub-fix wrote `merged.store_payloads |= f.store_payloads`,
    but Java's `FieldInfo.setStorePayloads` is
    ```java
    void setStorePayloads() {
      if (indexOptions.subsumes(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)) {
        storePayloads = true;
      }
      this.checkConsistency();
    }
    ```
    Without the guard, a source claiming payloads on a `Docs`/`DocsAndFreqs`
    field pushes the merged `FieldInfo` into `store_payloads = true,
    index_options = Docs` — a state `FieldInfo.checkConsistency` rejects
    ("indexed field cannot have payloads without positions"), i.e. a merged
    `.fnm` real Lucene's `FieldInfo` constructor refuses to load. Reachable
    whenever a caller hands `MergeSource` a hand-built `&[FieldInfo]`, which is
    exactly what the `IndexWriter`/FFI path does. **Fixed** — the OR is gated on
    a local `subsumes_positions` (mirroring `IndexOptions::subsumes_positions`,
    which is `pub(crate)` in `lucene-codecs` and deliberately not widened);
    test `store_payloads_or_respects_set_store_payloads_positions_guard` covers
    all three positionless options, `DocsAndCustomFreqs` included.

33. **[MISSING] No Java-generated ground truth for 1 700 lines of transliterated
    arithmetic.** The reviewer's point: 40 unit tests written from the same
    reading of `TieredMergePolicy.java` that produced the Rust cannot catch a
    shared misreading — a flipped comparison, a truncation in the wrong place, a
    level-walk off-by-one. `docs/parity.md`'s old "a pure heuristic decision
    function, so no Java fixture applies" reasoning does not survive contact with
    that argument. **Fixed** — new `fixtures/src/GenMergePolicy.java` +
    `fixtures/data/merge_policy/merge_policy.manifest.properties` +
    `crates/lucene-index/tests/merge_policy_fixtures.rs`:
    - The generator emits **no index** (a merge policy is a pure function of
      segment statistics) — it builds `SegmentInfos` by hand over a
      `ByteBuffersDirectory` with one real file per segment of exactly the
      requested byte length, so `SegmentCommitInfo.sizeInBytes()` and
      `MergePolicy.size()`'s pro-rating are genuinely exercised. Sizes convert
      to `setMaxMergedSegmentMB`/`setFloorSegmentMB`'s MB unit by dividing by
      2^20, exact in binary floating point, so they round-trip through Lucene's
      own `(long)(v * 1024 * 1024)` losslessly.
    - **33 scenarios**: 21 `findMerges` (empty, single, within-budget,
      below/above floor, no floor, 40-segment default tier, geometric mixed
      sizes, oversized excluded, oversized-but-delete-heavy included,
      deletes-over/under-budget, lowered `deletesPctAllowed`, the growth guard
      and its delete bypass, the currently-merging exclusion,
      `targetSearchConcurrency` 4 and 8, varying delete ratios, the byte cap
      binding before the merge factor, and equal sizes to pin the name
      tiebreak), 7 `findForcedMerges` (to 1/2/3/4, already-at-target,
      already-single, single-with-deletes), 5 `findForcedDeletesMerges`
      (none qualifying, mixed thresholds including the exact boundary,
      bounded-by-max-size, merging excluded, raised threshold).
    - The Rust test asserts the **exact** grouping including group order and
      within-group order — which pins `getSortedBySegmentSize`, the candidate
      construction order, and `findForcedMerges`' smallest-first bin walk, not
      just the membership. A second test asserts the fixture keeps covering
      every entry point and both outcomes, so it can't silently decay.
    - Deterministic (fixed segment ids, no `IndexWriter`), so
      `scripts/gen-fixtures.sh --check` compares it byte for byte; verified by
      generating twice.

    **All 33 scenarios matched on the first run.** That independently confirms
    findings 1–12: the constants, the budget walk, the score formula, the
    pro-rating, the sort order, the growth guard, the forced-merge bin packing,
    and the bounded forced-deletes merges.

34. **[MISSING] `FieldInfos.Builder.add`'s `putAttributes` was not ported.**
    Java does three things to an already-seen field: `verifySameSchema`,
    `putAttributes(fi.attributes())`, `setStorePayloads()`. Finding 18 ported
    the first and third; later sources' `attributes` were silently discarded in
    favour of the first-seen source's. **Fixed** — attributes are now
    `Map.putAll`'d (a later source's value wins for a shared key, its own keys
    are added); test
    `attributes_are_put_all_ed_across_sources_not_taken_from_the_first`.

35. **[MISSING] `reconcile_field_numbers` accepted a source naming the same
    field twice.** Java's `FieldInfos(FieldInfo[])` constructor rejects it, and
    so does this port's `field_infos::parse` — but `MergeSource::field_infos` is
    a caller-supplied slice that may be hand-built, and a duplicate silently
    lost one of the two field numbers when the map is inverted for the
    postings/points merge (finding 26b). **Fixed** —
    `Error::DuplicateFieldNameInSource`; test
    `a_source_naming_the_same_field_twice_is_rejected`.

36. **[INTENTIONAL, now documented] `merge_postings`/`merge_points` are
    deliberately stricter than Java for a *declared but empty* field.** Java's
    `MultiFields` tolerates a declared field whose reader returns no `Terms`,
    and `PointsWriter.mergeOneField` tolerates `values == null`. This port
    cannot tell that apart from "the caller forgot to open the `.tim`", because
    `SourcePostings`/`SourcePoints` are caller-supplied rather than pulled from
    a reader, and silently merging away a whole source's postings is the worse
    failure. The report and both error variants previously presented this as
    matching Java; it now says stricter-than-Java, with the reason and the
    condition under which to revisit.

37. **[INTENTIONAL, resolved] Dead `MergeType::ForceMerge` and an eight-argument
    `do_find_merges`.** Both were shape-parity-for-its-own-sake. The variant is
    deleted (with a doc note that Java declares but never uses it either), and
    the five budget values are bundled into a `MergeBudget` struct, which
    removes the `#[allow(clippy::too_many_arguments)]` and makes both call
    sites self-documenting.

38. **[INTENTIONAL, now documented] Three degenerate-input divergences and two
    extra scope boundaries.** Where Java's `x/0` yields `NaN` (which fails every
    `<=` comparison), the port substitutes a neutral value: `totalDelPct` and
    `segDelPct` become `0.0`, `nonDelRatio` becomes `1.0`, and `levelSize` is
    floored at 1. All need a zero-`doc_count` or zero-byte segment, which real
    Lucene never produces but this port's unit-agnostic `size_bytes` allows, and
    the substitutions are the conservative direction. Also documented: that
    `max_merge_at_once` is an extra knob Lucene 10 does not have (so setting it
    independently of `segments_per_tier` is outside anything Java can express),
    and — separately — the `lucene-ffi` entry point's doc now names all three
    unexposed knobs and warns that `reclaim_weight` changed meaning (it is now
    `score()`'s `Math.pow(nonDelRatio, N)` exponent; JVM callers should pass
    `2.0` for Lucene-identical scoring).

    The reviewer also flagged that most `find_merges` unit tests used
    `max_merge_at_once: 10, segments_per_tier: 2` — a config Lucene cannot be
    set to, since `mergeFactor` is derived and `setSegmentsPerTier` rejects
    `< 2.0`. Four tests (`max_merged_segment_size_caps_a_merge`, both
    growth-guard tests, `target_search_concurrency_bounds_merged_doc_count`)
    were rebuilt on Java-expressible configs where `max_merge_at_once ==
    segments_per_tier`, and now additionally assert that the intended cap — not
    the merge factor — is what bound. The integration test and
    `index_writer.rs`'s `tight_merge_policy()` keep the extra knob, but their
    behaviour is now pinned by the fixture suite instead.

---

## Carry-over items raised by this batch

- [ ] **Stored-fields bulk merge.** `lucene-codecs::stored_fields` needs a
      `MatchingReaders` equivalent, raw compressed-chunk access on the reader,
      and a writer entry point that accepts pre-compressed chunks (BULK) and
      pre-serialised documents (DOC), so `merge.rs` can stop
      decompressing→allocating→recompressing every document. Extends the
      existing b3 carry-over with the merge-side consequence. (Finding 24.)
- [ ] **Streaming postings merge.** A k-way-merged forward `TermsEnum` cursor
      plus a combined docs+freqs+positions read call in `lucene-codecs` would
      remove two dictionary seeks and one docs/freqs decode per (term, source)
      and the O(all postings) materialisation in `merge_postings`. Shares its
      blocker with the b4 carry-over A1 (lazy `SegmentTermsEnum`). (Finding 27.)
- [ ] **`BKDWriter.merge`.** A one-pass merge over already-sorted per-segment
      BKD trees, instead of re-indexing every live point. (Finding 29.)
- [ ] **`MergeSource` cannot carry per-source `min_version`/`has_blocks`**, so a
      merged `.si` claims the caller's version and `hasBlocks = false`.
      Unreachable while this port only merges segments it wrote itself; needs an
      exhaustive-struct-literal change across ~85 call sites plus
      `index_writer.rs`. (Finding 20.)
- [ ] **Zero-doc merges should be dropped by `IndexWriter::apply_merge`**, the
      way Java's `SegmentMerger.shouldMerge()` + `IndexWriter.commitMerge` do,
      rather than committing a real 0-doc segment. Belongs to b9/b11.
      (Finding 22.)
- [ ] **Sparse doc-values/norms merges** stay a hard error until
      `lucene-codecs` can write `IndexedDISI`-backed sparse fields. (Finding 21.)
