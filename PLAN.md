# lucene-rust: Porting Apache Lucene to Rust with FFI integration into OpenSearch

This is the master plan for porting Apache Lucene (Java, ~1.5M LOC in `lucene/core` alone)
to Rust, exposed over a JNI/FFI boundary so OpenSearch (JVM) can use it as a drop-in
engine for the hot paths. Source of truth for the Java side: `/home/tuong/work/lucene`.
OpenSearch checkout: `/home/tuong/work/OpenSearch`.

---

## 0. Strategy and non-goals

### Guiding decisions

1. **Port by on-disk format, not by class hierarchy.** The contract that matters is the
   Lucene index format (segments, postings, doc values, stored fields, points, vectors).
   If lucene-rust reads and writes bit-identical (or at least format-compatible) segments
   for one pinned codec version (e.g. `Lucene103`), Java Lucene and lucene-rust can
   coexist on the same index. This is the property that makes incremental adoption in
   OpenSearch possible.
2. **Pin one Lucene version.** Pick the version OpenSearch `main` ships (check
   `buildSrc/version.properties` → `lucene = ...`). Do not chase Lucene trunk during the
   port. Backward-codecs support is explicitly out of scope for v1; old segments are
   handled by the Java engine until force-merged.
3. **Read path first, write path second.** A read-only Rust searcher over
   Java-written segments delivers value early (query CPU is the OpenSearch hot path),
   is verifiable byte-for-byte against Java results, and avoids the hardest correctness
   risks (merge policy, deletes, transactional commits) until the foundations are proven.
4. **FFI = JNI via the `jni` crate + a thin handle-based C ABI.** OpenSearch is JVM;
   "FFI" concretely means a `cdylib` loaded by a JNI wrapper. Design the Rust-side API
   as a C ABI (opaque handles, no Rust types across the boundary) so the same library
   also works from Panama/FFM (`java.lang.foreign`), which is the better long-term
   binding (JDK 21+, which OpenSearch already requires).
5. **Differential testing is the correctness backbone.** Every milestone gates on
   comparing lucene-rust output against Java Lucene on the same input: same segments,
   same queries, same top-k docs and scores (within float tolerance), same term stats.

### Non-goals (v1)

- No port of: `luke`, `benchmark`, `demo`, `monitor`, `replicator`, `expressions`,
  `classification`, `spatial3d`, `spatial-extras`, `queryparser` (OpenSearch has its own
  query DSL; we accept pre-parsed query trees over FFI).
- No backward-codecs (old index versions).
- No index sorting, no soft-deletes semantics beyond what OpenSearch requires
  (OpenSearch **does** require soft-deletes for replication — this lands in Phase 6,
  it is required before write-path integration, just not before read-path integration).
- No scoring pluggability beyond BM25 + constant score + a similarity trait.

---

## 1. Architecture: crate layout

Cargo workspace, one crate per Java module boundary (roughly):

| Crate | Java counterpart | Contents |
|---|---|---|
| `lucene-util` | `o.a.l.util`, `o.a.l.internal` | BitSet/FixedBitSet, BKD utils, FST, packed ints, `BytesRef`, LSB radix sorter, SIMD kernels |
| `lucene-store` | `o.a.l.store` | `Directory`, `IndexInput/Output`, mmap (memmap2), buffered/NIOFS, checksums, locking |
| `lucene-codecs` | `o.a.l.codecs` | Codec trait + the one pinned default codec: postings (PFOR-delta), doc values, stored fields (LZ4/zstd blocks), points (BKD), KNN vectors (HNSW), norms, live docs, segment infos |
| `lucene-index` | `o.a.l.index` | SegmentReader/DirectoryReader, Terms/PostingsEnum, IndexWriter, DWPT, merge policy/scheduler, deletes, commits |
| `lucene-analysis` | `o.a.l.analysis` + `analysis/common` subset | TokenStream trait, StandardTokenizer (from Unicode segmentation), lowercase/stop/ascii-folding; everything else stays JVM-side long-term |
| `lucene-search` | `o.a.l.search` | Query/Weight/Scorer, Boolean (WAND/BMW), term/phrase/points ranges, collectors, BM25, ConstantScore, MatchAll |
| `lucene-core` | — | Facade crate re-exporting the above; the "public API" |
| `lucene-ffi` | — | `cdylib`: C ABI + JNI export layer, handle registry, panic → error-code mapping |
| `opensearch-plugin/` | — | Java: OpenSearch engine plugin (`EngineFactory`) + JNI binding class, native lib loading, CI packaging |

Rationale: matches Lucene's own dependency DAG (`util ← store ← codecs ← index ← search`),
lets phases parallelize, and keeps `lucene-ffi` as the only `unsafe`-heavy crate.

Key crates from the ecosystem to use rather than re-port: `memmap2` (mmap directory),
`zstd`/`lz4_flex` (stored fields), `crc32fast` (checksums), `unicode-segmentation`
(StandardTokenizer is UAX#29), `rayon` (concurrent merge/search), `jni` (JNI layer).
Study but do not depend on: **Tantivy** (license-compatible, MIT — prior art for nearly
every component; where a design question comes up, check how Tantivy solved it, but its
index format is NOT Lucene-compatible, which is exactly what we need to be).

---

## 2. Phases

### Phase 1 — Foundations: `lucene-util` + `lucene-store` (est. 6–8 weeks)

Port order within phase:

1. `BytesRef`/`BytesRefBuilder` → mostly `&[u8]`/`Vec<u8>` idioms; keep a thin newtype
   where ordering semantics (unsigned byte compare) matter.
2. `DataInput/DataOutput` primitives: vint/vlong, zigzag, `readGroupVInt` (group-varint —
   note Lucene 9.9+ uses this in postings), string (Java-modified-UTF8 **only** where the
   format demands; segment metadata uses standard UTF-8).
3. `Directory` + `IOContext`, `FSDirectory`, `MMapDirectory` (memmap2, madvise hints
   mirroring Java's `ReadAdvice`), `IndexInput` slicing/cloning — model clones as cheap
   offset-cursors over an `Arc<Mmap>`.
4. Checksums: `BufferedChecksumIndexInput` (CRC32), footer/header verification
   (`CodecUtil.checkHeader/checkFooter` — get the magic numbers and version framing exact).
5. `FixedBitSet`, `SparseFixedBitSet`, `LongBitSet`, `PackedInts`/`DirectWriter`/
   `DirectMonotonicReader` (doc values depend on these heavily — port with exhaustive
   round-trip tests against Java-generated fixtures).
6. FST reader (writer can wait until Phase 5): the terms index (`.tip`) is an FST.
   This is one of the two hardest data structures in the port (the other is BKD).
7. Locking: `NativeFSLockFactory` semantics via `flock`/`OpenOptions`.

**Test harness (build in this phase, use forever):** a Java "fixture generator" module in
this repo (`fixtures/`, Gradle, depends on the pinned Lucene) that writes: raw
packed-ints blobs, FSTs, and small single-segment indexes with known content. Rust tests
read fixtures and assert exact decoded values. Plus proptest round-trips for every
encoder we do implement.

**Exit criteria:** open a Java-written segment directory, verify all file checksums,
decode `segments_N` and `.si` files, dump the terms index of a text field.

### Phase 2 — Read-only codec: decode the pinned default codec (est. 10–14 weeks)

The heart of the port. For the pinned codec (e.g. `Lucene103Codec`), implement readers for:

1. **SegmentInfos / FieldInfos** (`segments_N`, `.si`, `.fnm`) — already started in P1.
2. **Postings** (`Lucene103PostingsFormat`: `.tim/.tip/.tie` terms dict + FST index,
   `.doc/.pos/.pay`): block-decoded FOR/PFOR (the `ForUtil`/`PForUtil` generated code —
   port the generator output, then vectorize with `std::simd` or explicit AVX2 behind
   feature flags; scalar fallback first), skip data (impacts!), `PostingsEnum` with
   positions/offsets/payloads.
3. **Impacts** (`ImpactsEnum`): required for WAND/MAXSCORE in Phase 3 — do not skip.
4. **Doc values** (`.dvd/.dvm`): numeric (direct-monotonic + gcd/table compression),
   sorted/sorted-set (term dicts + ordinals), binary, sorted-numeric. OpenSearch
   aggregations live on these — treat as first-class, not an afterthought.
5. **Stored fields** (`.fdt/.fdx/.fdm`): LZ4/zstd(actually DEFLATE in BEST_COMPRESSION
   mode — check pinned version) block decompression, prefetch-friendly random access.
6. **Norms** (`.nvd/.nvm`).
7. **Points / BKD** (`.kdd/.kdi/.kdm`): BKD tree reader + `intersect` visitor. Hard;
   needed for numeric/date range queries which OpenSearch uses constantly.
8. **Live docs** (`.liv`) and per-commit deletes/DV-updates generations.
9. **KNN vectors** (`.vec/.vex/.vem`, HNSW): schedule **last within the phase** and be
   willing to punt to Phase 8 — big, self-contained, and OpenSearch's k-NN often uses
   its own engines (faiss/nmslib) anyway.
10. `SegmentReader` + `DirectoryReader.open(commit)` + `MultiTerms`/leaf abstraction.

**Verification:** `CheckIndex`-equivalent in Rust (`lucene-rust check <dir>`), run against
indexes generated by the Java fixture generator from randomized documents (use Lucene's
own `RandomIndexWriter`-style randomization in the fixture generator, many codec
parameter combinations). Golden test: dump every posting, every doc value, every stored
doc from both sides and diff.

**Exit criteria:** Rust `CheckIndex` passes on randomized Java-written indexes including
deletes and DV updates; full-corpus dump diff is empty on a real dataset
(e.g. a Wikipedia sample indexed by Java Lucene).

### Phase 3 — Search: queries, scoring, collectors (est. 8–10 weeks)

1. Traits: `Query → Weight → Scorer/ScorerSupplier`, `DocIdSetIterator`,
   `TwoPhaseIterator`, `BulkScorer`. Use enums where the closed set allows
   (DISI is called per-doc — keep it monomorphizable; `Box<dyn>` only at Weight level).
2. Similarity: BM25 (exact same float math as Java — same order of operations, `f32`
   where Java uses float, precomputed norm cache tables) + constant score.
3. Queries, in order: `MatchAllDocs`, `TermQuery`, `BooleanQuery` (conjunction DISI,
   disjunction heap, minimum-should-match), `PointRangeQuery` (BKD intersect),
   `PhraseQuery` (exact + sloppy), `TermInSetQuery`, `PrefixQuery`/`WildcardQuery`
   (needs Levenshtein/automaton machinery — port `o.a.l.util.automaton` here; consider
   the `fst`/`regex-automata` crates for internals but keep Lucene semantics),
   `FunctionScore`-shaped hooks deferred.
4. Dynamic pruning: `WANDScorer`/block-max, `ImpactsDISI`, `MaxScoreCache`. This is
   where Lucene's search performance comes from; without it the port is not competitive.
5. Collectors: `TopScoreDocCollector` (with after/searchAfter), `TotalHitCountCollector`,
   early termination, `CollectorManager` + intra-query concurrency via rayon over leaves
   (mirror Lucene's leaf-slice model).
6. `IndexSearcher` facade + query cache (LRU on filter bitsets, like `LRUQueryCache`) —
   cache can be a later sub-milestone.

**Verification:** differential query harness — a Java CLI (in `fixtures/`) and Rust CLI
that both run a query file against the same index and emit `(docid, score)` top-1000;
diff with score tolerance 1e-5 relative and **exact** doc-set equality. Fuzz with
randomly generated boolean trees over randomized indexes. Also compare `explain()`-level
term stats for a sample.

**Exit criteria:** differential harness green over 100k randomized queries on randomized
indexes; luceneutil-style benchmark (wikimedium terms/phrases/booleans/ranges) shows
Rust ≥ Java on p50 and p99 for the ported query types.

### Phase 4 — FFI layer + read-only OpenSearch integration (est. 6–8 weeks, overlaps P3)

**C ABI design (`lucene-ffi`):**

- Opaque `u64` handles (generation-tagged slotmap) for: `Directory`, `IndexReader`,
  `IndexSearcher`, `Query`, prebuilt `TopDocs` result buffers. No Rust pointers cross
  the boundary; no callbacks from Rust into Java in v1 (collectors run entirely in Rust).
- All calls return `i32` status; results via out-buffers. `catch_unwind` at every entry
  point → error code + last-error message TLS slot. **A Rust panic must never unwind
  into the JVM.**
- Query representation across the boundary: a compact binary query tree (flatbuffer or
  hand-rolled tag-length-value — benchmark both; avoid protobuf/JSON per-query cost).
  OpenSearch already builds Lucene `Query` objects; we add a
  `QueryVisitor`-based serializer on the Java side for the supported subset, with a
  "unsupported → fall back to Java engine" escape hatch **per query**.
- Results: top-k `(doc, score)` + total hits written into a Java-owned direct
  ByteBuffer / MemorySegment to avoid JNI array copies. Stored-field fetch as a separate
  call (doc → bytes of the `_source` field).
- Two binding front-ends over the same C ABI: JNI (`jni` crate, works everywhere) and a
  Panama FFM `MethodHandle` layer (preferred at runtime on JDK 21+; measure — FFM
  downcalls are typically faster and avoid JNI-local-ref churn).

**OpenSearch plugin (`opensearch-plugin/`):**

- An `EnginePlugin` providing a custom `EngineFactory`. First deliverable: a
  **shadow-read mode** — the plugin opens the same shard directory read-only in Rust on
  each refresh (`DirectoryReader` handle refreshed on Lucene commit/refresh points),
  serves eligible search requests through Rust, everything else through the normal
  engine. Deletes visible via `.liv` per commit; near-real-time (in-memory) segments are
  NOT visible to Rust in this mode — acceptable for search-after-refresh semantics only
  if the shard is search-only/replica or `refresh` forces a commit; otherwise route
  NRT-sensitive requests to Java. Document this loudly.
- Native library packaging: per-platform `cdylib` (linux-x64/arm64 gnu, macOS arm64)
  inside the plugin zip, extracted and loaded at plugin init; crash-safety review
  (a segfault in Rust kills the whole node — this is why handle validation and no-raw-
  pointers matter).
- Benchmark with OpenSearch Benchmark (`nyc_taxis`, `pmc`, `big5` workloads), Rust vs
  Java engine on the same shards.

**Exit criteria:** an OpenSearch node serving term/bool/range/match queries for a real
workload through lucene-rust in shadow-read mode, with automatic per-query fallback,
and a benchmark report.

### Phase 5 — Write path: analysis chain + indexing (est. 12–16 weeks)

1. `lucene-analysis`: `TokenStream` as an iterator-of-token-structs (skip Java's
   AttributeSource reflection design entirely — a plain
   `Token { bytes, position_increment, offset, ... }` struct), StandardTokenizer via
   UAX#29 (`unicode-segmentation`), lowercase, stop, ASCII-folding.
   **Long-term stance:** analysis mostly stays on the JVM side in OpenSearch
   (analyzers are configured there, plugins provide them). So ALSO support
   "pre-analyzed" ingestion over FFI: Java runs the analyzer, ships tokens to Rust.
   This makes the Rust analysis chain a fast path, not a compatibility burden.
2. Codec **writers** for everything Phase 2 reads: postings writer (FOR/PFOR encode,
   skip/impacts writer), FST builder (hard — port `FSTCompiler` carefully; fixture:
   build FST from same term set in Java and Rust, require byte-identical output),
   doc values writers, stored fields (LZ4 fast mode first), points (BKD writer with
   offline sort for large fields), norms, `.si`/`segments_N`/`.fnm` writers, compound
   files (`.cfs/.cfe`).
3. Indexing chain: `IndexWriter`, DWPT-per-thread with in-memory hash (bytes → postings
   builder mirroring `BytesRefHash` + parallel arrays), flush-by-RAM accounting,
   `flush()` → segment.
4. Deletes/updates: delete-by-term/query queues, `BufferedUpdates`, frozen deletes
   applied on flush; doc-values updates can be deferred to 5b.
5. Commits: `SegmentInfos` two-phase commit (pending_segments_N → fsync → rename),
   `IndexFileDeleter` refcounting, `prepareCommit/commit/rollback` (OpenSearch translog
   recovery depends on 2-phase commit + commit user-data — must be exact).
6. Merging: `TieredMergePolicy` (port the math faithfully), `ConcurrentMergeScheduler`
   on a rayon/thread pool, merge readers reusing Phase 2, optimized bulk-merge paths
   (stored fields raw-chunk copy) later.
7. NRT: `DirectoryReader.openIfChanged(writer)` — reader from uncommitted flushed
   segments + in-memory deletes. Required for real OpenSearch refresh semantics.

**Verification:** the killer test — **cross-engine round-trip**: index corpus with Rust
→ open with *Java* Lucene → Java `CheckIndex` passes and Java search results match; and
the reverse. Then interleaved: Java writes segments, Rust merges them, Java reads the
result. Randomized crash-consistency tests (kill during commit, reopen, verify).

**Exit criteria:** Java `CheckIndex` clean on Rust-written randomized indexes;
cross-engine differential search green; sustained indexing throughput ≥ Java on
luceneutil `wikimediumall` ingest.

### Phase 6 — Full OpenSearch engine integration (est. 10–14 weeks)

1. Soft-deletes + `Lucene*SoftDeletesRetentionMergePolicy` equivalent — required for
   OpenSearch peer recovery / CCR-style retention leases.
2. Engine implementation: an `InternalEngine` alternative where IndexWriter lives in
   Rust — translog interplay (OpenSearch translog stays Java; Rust engine must expose
   sequence numbers, local checkpoint, commit user data exactly as
   `InternalEngine` does), refresh → Rust NRT reader, flush → Rust commit.
3. Segment replication mode (simpler than document replication for us: only primaries
   index; replicas use the Phase 4 read path) — recommend shipping this first.
4. `_source`, `_id`, `_seq_no`, `_primary_term`, `_version` field handling parity;
   get-by-id (term lookup on `_id`) fast path over FFI.
5. Aggregations: keep OpenSearch agg framework on Java initially, feed it via
   FFI doc-value cursors (batch columnar reads into shared buffers); native Rust
   terms/histogram/stats aggs as a follow-on performance phase.
6. Ops: memory accounting bridged to OpenSearch circuit breakers (Rust side reports RAM
   usage), stats APIs, slow log hooks, graceful shutdown, panic → shard-failed (not
   node-down) hardening where possible.

**Exit criteria:** OpenSearch integration test suite (`:server` engine tests adapted +
full REST test suite for search/index/get/delete) green on the Rust engine for a
supported feature matrix; multi-day soak test with random restarts, no index corruption.

### Phase 7 — Performance and SIMD hardening (continuous, dedicated 6–8 weeks)

- Vectorize: PFOR decode, dot-product/cosine (if vectors in scope), BKD compare loops,
  bitset ops — `std::simd` with runtime feature detection (AVX2/AVX-512/NEON).
- Profile-guided: flamegraphs vs Java async-profiler on identical workloads; close gaps.
- Memory: arena allocation in DWPT (Lucene's `ByteBlockPool` design translates well),
  `IOContext`-driven madvise, optional `io_uring` experiment for cold stored-field reads.
- FFI overhead budget: < 1µs per search call overhead; batch APIs wherever per-doc
  calls could occur.

### Phase 8 — Long tail (post-v1, prioritized backlog)

KNN/HNSW (if not done in P2), highlighting (needs term vectors — add `.tvd/.tvx` codec
support), suggesters (FST-based, reuse P5 FST builder), join/grouping/facets (OpenSearch
mostly reimplements these as aggs — likely never needed), index sorting, backward-codecs.

---

## 3. Cross-cutting engineering rules

- **Unsafe policy:** `unsafe` allowed only in `lucene-util` (SIMD, mmap access) and
  `lucene-ffi`; `#![forbid(unsafe_code)]` in all other crates. Miri on util tests.
- **Float discipline:** scoring must match Java: `f32` math in the same order; no FMA
  contraction in scoring paths (verify codegen); document every place we intentionally
  diverge.
- **Java-isms translation guide** (write `docs/porting-conventions.md` early):
  IndexReader lifecycle/refcounting → `Arc` + explicit `close` for mmap determinism;
  checked IOException → `thiserror` error enums per crate; `IndexInput.clone()` →
  cursor structs; ThreadLocal DWPT pools → per-thread slots keyed by rayon/thread id;
  Java unsigned-byte compares → `u8` slices (free win).
- **Fixture pinning:** the Java fixture generator pins the exact Lucene version; CI
  regenerates fixtures and runs the differential suites on every PR (Linux x64 + arm64).
- **Licensing:** this is a derivative work of Apache Lucene → Apache-2.0, keep NOTICE
  attribution.
- **Progress tracking:** a `docs/parity.md` matrix — every Java file in `core` mapped to
  ported / partial / not-needed / deferred, updated per PR.

## 3.5 Rust-first design: where we deliberately do NOT mirror Java

The on-disk **format** is the compatibility contract; the **in-memory design** is ours.
Rule of thumb: *port the bytes, not the objects.* Concretely:

1. **No GC-shaped object graphs.** Java Lucene's design is heavily driven by avoiding
   allocation/GC (ByteBlockPool, parallel arrays, AttributeSource reuse). In Rust we get
   deterministic memory for free, so: plain structs, arenas (`bumpalo`) per-DWPT and
   per-query where lifetimes are scoped, and struct-of-arrays layouts chosen for cache
   behavior — not to dodge a garbage collector.
2. **Monomorphization over virtual dispatch in per-doc loops.** Java pays a megamorphic
   call on every `DocIdSetIterator.nextDoc()`. We keep `dyn` only at Query/Weight level;
   scorers and DISIs are enums or generic so the per-doc loop inlines. Target: zero
   virtual calls inside `collect()` inner loops.
3. **Zero-copy reads end-to-end.** `IndexInput` over mmap yields `&[u8]` views;
   `BytesRef`-style copies only at true ownership boundaries. Stored fields / `_source`
   returned to FFI as borrowed slices into decompression buffers owned by the call
   context — never intermediate `Vec` churn like Java's `byte[]` copies.
4. **SIMD from the start, not as a retrofit.** Java's Panama vector code
   (`PanamaVectorUtilSupport`, generated `ForUtil`) fights the JIT; we write the PFOR /
   bitset / BKD-compare kernels once with `std::simd` + runtime dispatch
   (`is_x86_feature_detected!`), scalar fallback for correctness testing. The Java
   generated-code files are treated as *specs*, not sources to transliterate.
5. **Bounds checks engineered away, not ignored.** Hot decode loops operate on
   fixed-size arrays (`&[u8; 128*4]` blocks) or use iterator patterns the compiler can
   prove; `get_unchecked` only in `lucene-util` behind cargo-fuzz + Miri coverage.
6. **Thread model: ownership instead of synchronized.** Java Lucene is littered with
   locks/volatiles because everything is shared. We structure it as: immutable segment
   readers (`Arc`, lock-free), one owner per DWPT (no locking in the indexing hot path,
   channel-based handoff to flush), rayon leaf-slices for query concurrency. `Mutex`
   allowed only on control-plane state (commits, merge scheduling, deleters).
7. **io_uring / madvise as first-class citizens**, not JNI-gated afterthoughts:
   `IOContext` maps directly to `madvise` per file type (RANDOM for term dicts,
   SEQUENTIAL for merges, WILLNEED prefetch for BKD leaves), and the cold-read path is
   abstracted so an io_uring backend can drop in (Linux) without touching codecs.
8. **Async-free core.** No async runtime in the library — search/indexing are
   CPU-bound; blocking + rayon is simpler and faster. FFI callers get plain blocking
   calls (OpenSearch already dispatches on its own thread pools).
9. **Error paths off the hot path.** `Result` in APIs, but decode inner loops validate
   per-block (checksums, bounds) rather than per-value, so the happy path is
   branch-predictable.
10. **Skip Java's abstraction taxes entirely**: AttributeSource reflection (plain token
    structs), `IndexInput.clone()` object churn (Copy cursor structs over `Arc<Mmap>`),
    boxed `Integer`/autoboxing in collectors (never exists), `ThreadLocal` pools
    (scoped ownership), finalizers/`Cleaner` (Drop).

Each phase's exit criteria implicitly include: profile the ported component and confirm
it beats Java on the same workload *before* moving on — a slower "faithful" port is a
bug, and finding out early is the point of the phased structure.

## 4. Sequencing summary and effort

Rough serial-critical-path estimate (small senior team, 3–5 people who know both
Lucene internals and Rust): P1 (2mo) → P2 (3mo) → P3 (2.5mo, P4 overlaps) → P4 (1.5mo)
→ P5 (3.5mo) → P6 (3mo) → P7 (ongoing). **~14–18 months to a production-candidate
read+write engine**, with a demonstrable read-only OpenSearch speedup at ~7–8 months
(end of P4). The read-only milestone is the natural go/no-go checkpoint: if Rust search
isn't decisively faster there, stop before paying for the write path.

Biggest technical risks, in order:
1. **FST builder byte-compatibility** (P5) — mitigate: reader-only first, and consider
   accepting non-byte-identical-but-format-valid output (Java can read it; only golden
   tests need loosening).
2. **Two-phase commit / translog recovery semantics** (P6) — mitigate: segment
   replication first, exhaustive crash fuzzing.
3. **JNI crash blast radius** — a Rust bug can kill a node, not just a shard —
   mitigate: handle validation, fuzzing of the FFI surface, optional
   process-isolation mode (sidecar over shared memory) as a fallback design we keep
   sketched but don't build unless needed.
4. **Scoring drift** breaking top-k parity — mitigate: float discipline + differential
   harness from day one of P3.
