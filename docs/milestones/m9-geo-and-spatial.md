# M9 — Geo and spatial

> **Goal:** points and shapes on a sphere and a plane -- indexing, the
> queries, distance sorting and doc-values forms -- as Lucene has them, so
> OpenSearch's `geo_point` and `geo_shape` run natively.

| | |
|---|---|
| **Effort** | L |
| **Depends on** | [M7](m7-core-complete.md) (`document` fields, multi-dimension BKD writing) |
| **Unblocks** | native `geo_distance`, `geo_bounding_box`, `geo_shape`, geo sorting |
| **Status** | delivered 2026-10-03, all four criteria met (the real-world Tessellator corpus 2026-10-03): T9.1-T9.6 done. Performance caveats closed 2026-10-03: spatial-extras RPT intersects and heatmaps 1.17x-1.24x (quad cells as values, `docs/parity.md`); the geo-filtered `terms` aggregation 1.30x query phase / 1.03x REST (the plugin's `terminated_early` replay, T9.6 benchmark). A `terms` aggregation behind a *dense* non-geo filter, 0.81-0.98x in the T9.6 benchmark, closed 2026-10-05 in process: 1.41x-2.97x (`scripts/bench-micro.sh --bench aggs`, below) |

---

## Why this milestone exists

`lucene-core`'s `geo` package (28 files, 5.7k lines) and the `LatLon*`/`XY*`
fields and queries in `document` have no rows in [`parity.md`](../parity.md);
20 of the 28 unreferenced core queries are geo. OpenSearch maps `geo_point`
and `geo_shape` straight onto them, so every geo query falls back to Lucene
today. `spatial3d` (106 files, 16.7k lines) and `spatial-extras` (76 files,
7.7k lines) complete the set; OpenSearch uses neither directly, so they come
last within the milestone.

---

## Scope

### In scope

- `geo`: `GeoEncodingUtils`, `Polygon`/`Line`/`Circle`/`Rectangle`,
  `Component2D` and its trees, the `Tessellator`, `XY*` geometry.
- `document`: `LatLonPoint`, `LatLonDocValuesField`, `LatLonShape`,
  `XYPointField`, `XYShape`, `ShapeField`'s triangle encoding, and every query
  and sort they build (`LatLonPointDistanceQuery`, `…InPolygonQuery`,
  `…DistanceFeatureQuery`, `LatLonShapeQuery`, the doc-values forms, the
  distance comparator).
- `spatial3d` and `spatial-extras`.
- OpenSearch: `geo_point` and `geo_shape` queries and `geo_distance` sorting
  as native shapes.

### Out of scope

- OpenSearch's geo aggregations (`geohash_grid`, `geotile_grid`,
  `geo_bounds`); they are aggregation work for a later plugin milestone.

---

## Tasks

- **T9.1** — `geo` primitives and the `Tessellator`, differential against
  Lucene on random polygons (triangle for triangle).
  **Done (2026-10-02).** All 27 `geo` classes plus `util/SloppyMath` are
  ported into `lucene-util` (`geo/`, `sloppy_math.rs`, and `strict_math.rs`
  for the fdlibm `StrictMath` functions SloppyMath's tables are built from);
  `PointValues.Relation` moved down to `lucene-util` with a re-export from
  `lucene_codecs::points`. Differential: `GenGeo`, `GenGeoTessellator`,
  `GenGeoParsers` -> `crates/lucene-util/tests/geo_fixtures.rs` -- bit-exact
  math and encodings, every `Component2D` query over ~100 shapes, the
  Tessellator triangle for triangle (or failure for failure) over ~200
  polygons incl. holes, poles, the dateline, self-intersections, the morton
  and SPLIT paths, and both parsers' results and error messages/offsets.
  One documented inexactness: `Rectangle.axisLat` goes through HotSpot's
  `Math.cos` intrinsic, which no portable code reproduces bit for bit (within
  2 ulps; exact on >99% of the corpus). Benchmark pair `scripts/bench-micro.sh
  --bench geo`; ratios in `docs/parity/util-geo.md`.
- **T9.2** — `LatLonPoint`/`XYPoint`: fields, queries, distance sort.
  **Done (2026-10-02).** Fields in `lucene-index/src/document/geo.rs`
  (`LatLonPoint`, `LatLonDocValuesField`, `XYPointField`,
  `XYDocValuesField`); queries, sorts and `nearest` in
  `lucene-search/src/document/geo/`: `LatLonPoint.newBoxQuery` (dateline
  split), `LatLonPointDistanceQuery` (forward and inverse walks),
  `LatLonPointQuery` under all four relations through `SpatialQuery`'s
  sparse/dense/inverse/contains scorers and `hasAnyHits`,
  `XYPointInGeometryQuery`, the three doc-values queries,
  `LatLonPointDistanceFeatureQuery` (scores and `setMinCompetitiveScore`
  pruning), `LatLonPointSortField`/`XYPointSortField` with their
  comparators' bounding-box `compareBottom` (also as `CUSTOM` keys of
  `search_sorted`), and `NearestNeighbor` (best-first over a lazily navigated
  `PointTreeNode`, `java.util.PriorityQueue`'s tie order). Supporting codec
  changes: a multi-dimension leaf is related by its own stored box before it
  is decoded (Java's `visitDocValuesWithCardinality`), and a singleton
  `SORTED_NUMERIC` value is read through the resolved values array.
  Differential: `GenGeoPoints` -> `crates/lucene-search/tests/geo_points_fixtures.rs`,
  ~1100 queries (hits, score bits, sort-value bits, distance bits) equal over
  Lucene's index and over this port's index of the same documents (whose
  packed points are compared byte for byte too); write path:
  `VerifyGeoPoints` replays them through Lucene over the Rust-written index.
  Benchmark pair `scripts/bench-micro.sh --bench geo_points`; ratios in
  `docs/parity.md`. `SpatialQuery`'s shape half stays with T9.3.
- **T9.3** — Shapes: `LatLonShape`/`XYShape` writing and querying, including
  the doc-values shape encoding.
  **Done (2026-10-02).** `ShapeField`'s triangle encoding, the
  `LatLonShape`/`XYShape` field factories and the shape doc values
  (`ShapeDocValues`' tree, byte for byte, and its `relate`) in
  `lucene-index/src/document/{shape.rs, shape_doc_values.rs}`; the queries
  in `lucene-search/src/document/geo/shape_queries.rs` -- `LatLonShapeQuery`,
  `XYShapeQuery` and `LatLonShapeBoundingBoxQuery` through `SpatialQuery`'s
  scorers (now shared by points and triangles), `EncodedRectangle`, and the
  doc-values queries -- with the `lat_lon_shape`/`xy_shape` factories in
  `geo/mod.rs`. `StrictMath.hypot` joined `strict_math.rs` (the doc value's
  centroid weights lines by `Math.hypot`). Differential: `GenGeoShapes` ->
  `crates/lucene-search/tests/geo_shapes_fixtures.rs`: 1576 queries, every
  geometry under every relation, indexed and doc-values, lat/lon and
  cartesian, answered as Lucene answers over Lucene's index and over this
  port's (whose triangles and doc values are Lucene's byte for byte); 1500
  triangles encoded/decoded; 140 doc values and 700 `relate`s. Write path:
  `VerifyGeoShapes`. Benchmark pair `scripts/bench-micro.sh --bench
  geo_shapes`: every case above 1.0 (1.09x-1.76x); ratios and the stage-3
  changes in `docs/parity.md`.
- **T9.4** — `spatial3d`. *Done (2026-10-02).* All 103 classes of
  `lucene-spatial3d` (`docs/inventory/lucene-spatial3d.tsv`,
  `check-port-inventory.py --module spatial3d`, in the gate). The `geom`
  package (94 classes) is in `crates/lucene-util/src/spatial3d/` beside
  `geo` (pure math); `Geo3DPoint`/`Geo3DDocValuesField` in
  `lucene-index/src/document/geo3d.rs`; `Geo3DUtil`,
  `PointInGeo3DShapeQuery` and its visitor in
  `lucene-search/src/document/geo/geo3d.rs`; the two sort fields and
  comparators in `lucene-search/src/document/geo/sort.rs`. Differential:
  `GenGeo3d` -> `geo3d_fixtures.rs` (every shape kind on five planet
  models: serialized bytes both ways, bounds, membership, distances,
  relationships against shapes and x/y/z solids), `GenGeo3dMath` ->
  `geo3d_math_fixtures.rs` (the primitives' public API, and
  `StrictMath.tan/atan/atan2`, now in `strict_math.rs`), all bit for bit;
  `GenGeo3dPoints` -> `geo3d_points_fixtures.rs` (every query factory and
  every sort, over a four-segment index with deletions and a 12 000-point
  segment, on Lucene's index and on this port's: same hits, same sort-value
  bits), and `VerifyGeo3D` in `scripts/verify-write-path.sh` (Lucene reads
  the Rust-written index: `CheckIndex`, then every query answered as over
  its own). The generators run with HotSpot's trig intrinsics off; what
  that changes against a stock JVM is measured in `docs/parity.md`. One
  deliberate difference: Lucene's two geo3d comparators mis-read
  multi-valued documents in `copy()` (a Lucene bug, `docs/parity.md`); the
  port reads each document's own values. Benchmark pairs `--bench geo3d`
  and `--bench geo3d_points`: 1.10x-1.74x, but `geo3d_distance` (0.97x
  and 1.05x in two runs, inside the noise floor both times; written up in
  `docs/parity.md` with the ratios).
- **T9.5** — `spatial-extras` (prefix trees, `SpatialStrategy`s).
  *Done (2026-10-03).* All 65 classes of the jar (`check-port-inventory.py
  --module spatial-extras`). **Dependency decision:** spatial-extras is built on
  Spatial4j 0.8 and s2-geometry-library-java 1.0.0, third-party Java
  libraries with no Rust equivalent; the subsets it exercises are ported as
  faithful ports in `lucene-util` (`spatial4j/`, `s2/`, beside `geo/` and
  `spatial3d/`), differentially tested against the real jars, both
  Apache-2.0 (`docs/licences.md`, `NOTICE`). JTS (Spatial4j's optional
  polygon backend) is not ported: Lucene does not ship it, and
  spatial-extras makes polygons through Geo3D. Ported: the Spatial4j
  and S2 subsets and Lucene's Geo3D bridge
  (`lucene-util/src/spatial_extras/spatial4j.rs`), `GenSpatial4j` ->
  `spatial4j_fixtures.rs` (12 577 records); the
  `check-port-inventory.py --module spatial-extras` gate
  (`docs/inventory/lucene-spatial-extras.tsv`); the prefix trees (quad,
  packed quad, geohash, S2 at arities 1-3, number-range and date-range, with
  the `GregorianCalendar` subset the date tree needs) and the `query`
  package (`lucene-util/src/spatial_extras/{prefix_tree,query.rs}`),
  `GenSpatialPrefixTree` -> `spatial_prefix_tree_fixtures.rs` (4 534
  records: byte-identical cell tokens for every tree, date parse/format
  round trips, `SpatialArgsParser`); every strategy
  (`lucene-search/src/spatial/`: RPT, term-query, number-range, BBox,
  point-vector, serialized doc values, composite, heatmaps, date facets, the
  value sources), `GenSpatialStrategies` -> `spatial_strategies_fixtures.rs`
  (byte-identical fields, 1 901 answers over Lucene's index and this port's),
  and `VerifySpatialExtras` in `scripts/verify-write-path.sh` (Lucene runs
  `CheckIndex` on the Rust-written index and answers every question over it
  as over its own). Benchmark pair `--bench spatial_extras`: date ranges
  1.31x, BBox with the overlap-ratio similarity and RPT indexing a Geo3D
  polygon 0.98x/0.99x (inside the noise floor in both runs), RPT
  intersects 0.74x-0.77x, heatmaps 0.40x -- the last two left with their
  cause written up in `docs/parity.md`: a sorted stream of `seekCeil`s
  that this port's terms enum restarts at the trie's root each time
  (`c1-lazy-blocktree.md` F-9), and per-cell allocation. *Closed 2026-10-03:* a third stage-3 round made the quad
  tree's cells plain values -- `QuadCellRelater` relating a cell from its
  token bytes with one reused rectangle, `visit_quad` traversing query cells
  without a boxed cell or iterator per node, a haversine circle's
  `contains` precomputed -- every answer unchanged: RPT intersects 1.24x
  (rectangles) and 1.17x (circles), heatmaps 1.23x (`docs/parity.md`). The review closed
  four ways corrupt bytes could panic or abort: quad/packed-quad terms
  deeper than the tree, a date term's year overflow, an S2 term with no
  level, and unbounded nesting in binary shapes, geo3d streams and WKT.
- **T9.6** — Plugin wiring for OpenSearch's geo queries and sort.
  *Done (2026-10-03).* What OpenSearch 3.8.0 builds, mapped from its sources
  (`GeoBoundingBoxQueryBuilder`, `GeoDistanceQueryBuilder`,
  `GeoPolygonQueryBuilder`, `GeoShapeQueryBuilder`, the
  `VectorGeoPointShapeQueryProcessor`/`VectorGeoShapeQueryProcessor` pair,
  `GeoDistanceSortBuilder`): on a `geo_point`, `LatLonPoint.newBoxQuery` /
  `newDistanceQuery` / `newPolygonQuery` inside an `IndexOrDocValuesQuery`
  with the `LatLonDocValuesField` twin; on a `geo_shape`,
  `LatLonShape.newGeometryQuery` (`LatLonShapeQuery`,
  `LatLonShapeBoundingBoxQuery`, a `BooleanQuery` of them for a `CONTAINS` of
  several geometries or across the dateline) inside a `ConstantScoreQuery`,
  every relation, inline or `indexed_shape` alike; the `_geo_distance` sort
  is Lucene's `LatLonPointSortField` for one origin, metres, ascending, `min`,
  and OpenSearch's own comparator source otherwise (`MultiValueMode` over
  `GeoDistance.calculate` of every value and origin). All of it now runs
  natively: `GeoEncoder.java` reads the queries (the four package-private
  classes through `Reflect`) into query-tree nodes 15-19 (ABI 30), decoded
  through the validating constructors and run as `DocumentClause` leaves of
  the scorer tree keyed for the query cache by their own bytes; the sorts
  are sort key 7 (ABI 31), Lucene's comparator with its bounding-box
  `compareBottom`, OpenSearch's as `OpenSearchGeoDistanceSort`. Falls back by
  name: `search_after` on Lucene's geo sort (`search_after_geo`; its
  `compareTop` compares metres), `distance_type: plane` in OpenSearch's
  comparator (`sort_geo_plane`: `Math.cos`'s HotSpot intrinsic), nested geo
  sorts, and the deprecated prefix-tree `geo_shape` mapping
  (`clause_IntersectsPrefixTreeQuery`). Proved by `NativeSelfTest.geo` (704
  random geo queries over points and shapes at the poles and the
  antimeridian, degenerate polygons included, and ~900 sorted pages with a
  geo key, against Lucene in process; seen to fail on a radius off by 0.1%,
  a cache key that ignored the geometry and a wrong even median) and by
  `geo_matrix.py` in `scripts/verify-opensearch.sh` (90 request shapes on a
  one- and a three-shard index and on a merged one past 10,000 documents,
  against a stock node). New fuzz target `jvm_search_sorted`; geo seeds for
  `jvm_search`. Benchmark below.

### T9.6 benchmark: native geo searches against a stock node

`opensearch-plugin/e2e/phase_bench.py` with `GEO=1` (the geo rows of
`geo_matrix.py`, 93 native shapes) on one shard of 100,000 documents of
points and shapes (`GEO_LOAD=100000`: six segments of ~16,000), the index
switched between the native path and Lucene six times. By the plugin's
query-phase counters (Lucene over native): median 1.88x, two shapes under
1.0 -- `terms` aggregation behind a `geo_distance` filter (0.68-0.80x; the
same aggregation behind a `match` measured 0.93x on this node, and the geo
filter alone 1.93x) and `_geo_distance` across the antimeridian (0.94x,
then 1.10x on a re-run: noise). *The `terms` row, closed 2026-10-03:* not
the aggregation. JFR on the node put most of that query phase in Lucene's
`LatLonDocValuesQuery.createWeight` (its `createComponentPredicate` grid)
called from the plugin's `countTerminatedEarly`: under concurrent search a
`size: 0` count past `track_total_hits` (12,000 hits against the default
10,000) asks Lucene's weight which segments `Weight.count` answers and then
replays the count natively -- a second search. Now the aggregations' pass
hands back each segment's match count (ABI 32) and the plugin replays from
those, skipping the weight whenever no slice could stop even iterating
everything (the answer only grows with the iterated segments). On the same
100,000-document index, query phase / REST, Lucene over native: behind
`geo_distance` 0.62x -> **1.30x** / 0.82x -> **1.03x**, behind a
`bool` filter 0.63x -> 1.18x / 0.83x -> 1.12x, and with
`track_total_hits: 15000` (no replay at all) 1.43x / 1.16x. Left below 1.0,
measured: the same aggregation behind a *dense* match -- a `match` of
41,624 hits 0.93x / 0.98x, a `range` of 59,000 0.81x / 0.84x -- where the
native pass collects the matches into a document list, marks them in a bit
set and streams the ordinal column against it (three passes; the node's
profile puts 11% of its CPU in the first), where OpenSearch's collector
counts each match's ordinal in one; the slices there pass 10,000 hits, so
Lucene's weight is still asked (cheap for a term or a range). *Closed 2026-10-05,
in process* (`scripts/bench-micro.sh --bench aggs`: `AggsMicro.java`, the
`DenseGlobalOrds` collect loop and `buildAggregations` over Lucene, against
`micro_aggs.rs`, the native `aggregate_sliced_counting`; one segment of the
benchmark corpus, a frequent and a mid-frequency term and a 70% range, `keyword`
and `cat`): the native pass now marks the matches as it collects them (a
constant-scored bit set -- a points range, a cached clause -- copied a word
at a time; a postings batch's bits gathered in a register), and counts the
ordinal column a 1,024-document window at a time into per-segment counts by
segment ordinal (a small dictionary's spread over four counter lanes),
mapped to global ordinals once per segment. Before 0.35x-0.97x, after
1.41x-2.97x (pinned, three interleaved repetitions, noise floor 1.10x). The
REST benchmark was not re-run. Over REST round trips (`REST=1`): median 1.06x; the rows
under 1.0 are cheap `geo_shape` queries that both engines answer from
their query caches (OpenSearch wraps `geo_shape` in a `ConstantScoreQuery`),
where the response is ~2 ms and the query phase 40-120 us: there the native
call's fixed cost shows -- in one JVM a cached geo filter costs 20 us
natively against 7-15 us in Lucene, the same 18-20 us floor a cached term
filter pays natively (the geo node's own share, after the key fix, is under
2 us). Uncached, in process on the same shard: point box 0.72x
(`document::PointRangeQuery`, not the plugin's points path) / distance
1.87x / polygon 1.47x; shapes 0.99-1.25x (intersects box 0.99x, within box
1.25x, disjoint 1.13x, contains line 1.23x); Lucene's distance sort 1.28x,
OpenSearch's 1.75x. What closed the first measurement's gaps (sorts 0.5-0.7x,
empty `CONTAINS`/`WITHIN` filters 0.06-0.19x): Lucene's `compareBottom`
through a new comparator hook, the points read from the segment's decoded
column, and empty answers kept in the query cache, as `LRUQueryCache` keeps
`DocIdSet.EMPTY`.

## Acceptance criteria

- [x] Every geo query returns the same hits as Lucene on a generated corpus
      of random points and shapes, including the antimeridian, the poles and
      degenerate polygons. *Evidence:* `GenGeoPoints` (~1,100 queries) and
      `GenGeoShapes` (1,576; shapes on the poles and the dateline, slivers
      and collinear runs) against Lucene's index and this port's
      (`geo_points_fixtures.rs`, `geo_shapes_fixtures.rs`), the geo3d and
      spatial-extras generators likewise, and through the plugin
      `NativeSelfTest.geo` (704 random queries, degenerate polygons -- on one
      parallel or meridian, repeated vertices -- included).
- [x] The `Tessellator` produces Lucene's triangles, or fails where Lucene
      fails, on a corpus of real-world polygons. *Evidence:*
      `GenGeoTessellatorReal` -> `geo_fixtures.rs`'
      `tessellator_matches_lucene_on_real_world_polygons` over
      `fixtures/corpus/real_polygons.z` (`scripts/gen-tessellator-corpus.py`):
      Lucene 10.5.0's own `TestTessellator` shapes (21 resource files, 52
      inline polygons from issue reports) and Natural Earth v5.1.2 -- 1:50m
      countries, provinces and lakes, 1:10m lakes with islands and ten 1:10m
      countries (Russia's 22,908-point mainland split at the antimeridian,
      Antarctica at the pole, Norway's and Chile's coastlines, enclaves as
      holes): 1,111 shapes, 4,193 polygons, ~399,000 vertices, each parsed by
      both engines' parsers and tessellated with and without
      `checkSelfIntersections` (Lucene's shapes as cartesian polygons too):
      833,628 triangles identical (digest per polygon, full lists for
      `TestTessellator`'s shapes) and all 33 failures identical, messages
      included. Speed on the same corpus: `tessellate_real` 1.11x,
      `tessellate_real_checked` 1.08x (`--bench geo`). The ~200 seeded
      synthetic polygons of `GenGeoTessellator` stay beside it.
- [x] Real Lucene reads Rust-written point and shape indices.
- [x] Native `geo_distance` and `geo_shape` searches agree with a stock node
      in the plugin's matrix, and are no slower than Lucene. *Evidence:*
      `scripts/verify-opensearch.sh --docs 20000 --yaml` (2026-10-03):
      9,461 checks against a stock node, 0 failures, 95 geo request shapes
      on a one- and a three-shard index and five rounds on a merged
      12,000-document segment, every geo query and `_geo_distance` sort row
      native but the four that fall back by name; OpenSearch's YAML suites
      fail identically with and without the plugin (4 of 501). Speed: the
      T9.6 benchmark above -- by the query-phase counters median 1.88x
      Lucene, every geo query and sort row at or above 1.0 beyond noise; the
      one row below, a `terms` aggregation behind a geo filter, is now
      1.30x (query phase) / 1.03x (REST): the gap was the plugin's
      `terminated_early` replay building Lucene's geo weight, not geo
      (T9.6 benchmark). Over
      REST median 1.06x; cheap `geo_shape` queries answered from the query
      cache sit at 0.85-0.99x there, the native call's fixed ~10 us more
      than Lucene's cached path (written up above), not geo work.

## Risks and unknowns

- **Floating-point geometry.** Encoding and relation tests are
  quantization-sensitive; the port must follow Lucene's arithmetic exactly,
  or edge points flip in and out of results.

## Exit artifacts

- Geo fixture generators and the real-world polygon corpus
- `docs/parity.md` rows for `geo`, the geo `document` classes, `spatial3d`,
  `spatial-extras`

## Ledger evidence

The fixtures and benchmarks behind the parity ledger's rows for this
milestone, by ledger file (current facts; the rows carry the status).

### [spatial-extras.md](../parity/spatial-extras.md): lucene-util / lucene-search -- spatial-extras (M9 T9.5)

**Ported**: every class of the jar
(`docs/inventory/lucene-spatial-extras.tsv`). spatial-extras is built on two
third-party Java libraries with no Rust equivalent: **Spatial4j 0.8** (its
shape model -- `SpatialContext`, `Shape`/`Point`/`Rectangle`/`Circle`,
`SpatialRelation`, `DistanceCalculator`, the WKT reader, `BinaryCodec`) and
**s2-geometry-library-java 1.0.0** (`S2PrefixTree`'s cell ids). Decision
(T9.5): port the subset spatial-extras exercises, differentially tested
against the real jars (both Apache-2.0: `docs/licences.md`, `NOTICE`), in
`lucene-util` beside `geo`/`spatial3d`. JTS is not ported (Lucene does not ship
it; polygons go through Geo3D, `Geo3dShapeFactory`); a non-Geo3D context
answers a POLYGON with Spatial4j's `UnsupportedOperationException`, as Java
without JTS.

Tests, all generated with HotSpot's trig intrinsics off (as `GenGeo3d`):
- `fixtures/src/GenSpatial4j.java` -> `spatial4j/spatial4j.tsv` ->
  `crates/lucene-util/tests/spatial4j_fixtures.rs`: eight contexts (geodetic
  haversine/law of cosines/Vincenty, wrapping, planar bounded/unbounded,
  Geo3D sphere/WGS84), 160 random shapes each, relations, distances,
  `pointOnBearing`, binary codec both ways, WKT (results and errors),
  `DistanceUtils`, geohashes, S2 cells: 12 577 records bit for bit (a NaN's
  sign aside).
- `fixtures/src/GenSpatialPrefixTree.java` -> `spatial_prefix_tree/trees.tsv`
  -> `crates/lucene-util/tests/spatial_prefix_tree_fixtures.rs`: 14 trees
  (geohash, quad, packed quad pruned or not, S2 arity 1-3, planar and Geo3D),
  cells of 70 shapes per tree, terms read back, both date trees,
  `SpatialArgsParser`/`SpatialOperation`: 4 534 records.
- `fixtures/src/GenSpatialStrategies.java` (+ `SpatialExtrasCorpus.java`) ->
  `spatial_strategies/` -> `crates/lucene-search/tests/spatial_strategies_fixtures.rs`:
  400 documents through 15 strategies, four segments with deletions; every
  field token byte for byte, then 1 901 answers (every strategy x operation,
  BM25 scores for RPT's point term query, value sources bit for bit,
  heatmaps, date facets, `toString`) over Lucene's index and this port's.
  `VerifySpatialExtras` (`scripts/verify-write-path.sh`) has Lucene check and
  query the Rust-written index. `arbitrary_terms_read_without_panicking`:
  3 000 random terms per tree, errors never panics.

Bench: `scripts/bench-micro.sh --bench spatial_extras --reps 5`
(`SpatialExtrasMicro.java` / `micro_spatial_extras.rs`, 100 000 docs, 11-level
quad RPT; 2026-10-03, noise 1.17x): `spx_rpt_intersects_rect` 1.24x,
`spx_rpt_intersects_circle` 1.17x, `spx_heatmap` 1.23x, `spx_date_range`
1.38x, `spx_bbox_similarity` 1.08x, `spx_rpt_index_polygon` 1.01x (last two in
noise). These need the quad cells as values (`QuadCellRelater`, `visit_quad`,
`visit_scanned_quad`, `Cell::rect_bounds`, `HaversineWithin`), the block
tree's seek-state reuse and block-at-a-time `collectDocs`; the earlier
profiles are in git history and `docs/sweep/m2/c1-lazy-blocktree.md` F-9.

### [spatial3d.md](../parity/spatial3d.md): lucene-util / lucene-index / lucene-search -- spatial3d (M9 T9.4)

`lucene-spatial3d`'s `geom` package (shapes on an
ellipsoid as planes through x/y/z space: relations, bounds, distances, stream
format) in `lucene-util` beside `geo`, plus `Geo3DPoint` and its query/sorts.
No index format of its own, but `Geo3DPoint` indexes the x/y/z these compute
and `PointInGeo3DShapeQuery` asks `GeoArea.getRelationship` per BKD cell, so
everything is held to Java's bits.

Tests (two seeded corpora, record for record): `fixtures/src/GenGeo3d.java` ->
`geo3d/shapes.tsv` -> `crates/lucene-util/tests/geo3d_fixtures.rs` (five planet
models; every factory branch and shape kind incl. degenerate, polar,
antimeridian and untileable polygons with holes; per shape the
`writePlanetObject` bytes or exception and the round trip, bounds, edge points,
membership, distances per style, relationships vs other shapes and x/y/z
solids) and `fixtures/src/GenGeo3dMath.java` -> `geo3d/math.tsv` ->
`crates/lucene-util/tests/geo3d_math_fixtures.rs` (`Vector`, `GeoPoint`,
`Plane`, `SidedPlane`, `PlanetModel` + `DocValueEncoder`, `XYZBounds`,
`LatLonBounds`, `DistanceStyle`). All equal bit for bit and message for
message except a `NullPointerException`'s text (compared by class).
Twenty-four extra seeds (`-Dgeo3d.seed=N`, test `GEO3D_FIXTURES=dir`) were
swept once, all equal.

**Trigonometry.** HotSpot's `Math.sin/cos/tan` intrinsics differ from fdlibm in
the last bit for ~3% of arguments; the port uses fdlibm and the generators run
with `-XX:DisableIntrinsic=_dsin,_dcos,_dtan` (`scripts/gen-fixtures.sh`).
Against a stock JVM 5 824 of 50 571 records change in last bits; of 39 608
membership/relationship answers none changes for the same input.

Bench (2026-10-02, cross-engine digests): `--bench geo3d` (`Geo3dMicro.java` /
`micro_geo3d.rs`): `geo3d_polygon_build` 1.41x, `geo3d_within` 1.20x,
`geo3d_relate` 1.18x, `geo3d_circle_build` 1.10x, `geo3d_distance` 1.01x
(T9.4 review, 9 reps, noise 1.10x; needs concrete-typed plane bounds,
`intersection_coords`, a signum-free `isWithin`,
`Plane::find_intersections_two`/`find_intersections_arr`, `strict_math::sin_cos`).
`--bench geo3d_points` (`Geo3dPointsMicro.java` / `micro_geo3d_points.rs`,
300 000 WGS84 points): `geo3d_query_path` 1.74x, `geo3d_query_box` 1.69x,
`geo3d_query_distance` 1.65x, `geo3d_query_polygon` 1.42x,
`geo3d_distance_sort` 1.26x, `geo3d_outside_sort` 1.18x.

### [geo-shapes.md](../parity/geo-shapes.md): lucene-index / lucene-search -- geo shapes (M9 T9.3)

The shape half of Lucene's geo `document` classes:
`ShapeField`'s triangle encoding, the `LatLonShape`/`XYShape` fields and their
shape doc values, every shape query (`SpatialQuery`'s visitors under all four
relations, the encoded-space bounding-box query, the doc-values queries).

Tests: `fixtures/src/GenGeoShapes.java` (answers through `VerifyGeoShapes.run`;
package-private doc-values constructors through `fixtures/src/ShapeAccess.java`)
-> `crates/lucene-search/tests/geo_shapes_fixtures.rs`: 1576 queries
(polygons with holes, lines, points, circles, boxes across the dateline and on
the poles, multi-geometries; all four relations; indexed and doc-values;
lat/lon and cartesian) over a four-segment index with deletions, answered
identically (hits, score bits) over Lucene's index **and** this port's, whose
triangles and doc values equal Lucene's byte for byte; plus 1500 triangles
through `encodeTriangle`/`decodeTriangle` and 140 doc values (bytes, header,
centroid, bounding box, 700 `relate` answers). Write path:
`scripts/verify-write-path.sh` runs `write_geo_shapes_fixture` ->
`VerifyGeoShapes`.

Bench: `scripts/bench-micro.sh --bench geo_shapes` (`GeoShapesMicro.java` /
`micro_geo_shapes.rs`, 200 000 shapes; 2026-10-02, noise 1.07x):
`shape_index_fields` 1.09x (almost all Tessellator cost), `shape_index_doc_value`
1.63x, `shape_intersects_polygon` 1.55x, `shape_within` 1.24x,
`shape_contains_point` 1.76x, `shape_doc_values_box` 1.53x (needs `visit_many`
once per run and the fixed-window header decode).

### [geo-points.md](../parity/geo-points.md): lucene-index / lucene-search -- geo points (M9 T9.2)

The point half of Lucene's geo `document` classes: the
four fields and every query, sort and `nearest` they build.

Tests: `fixtures/src/GenGeoPoints.java` ->
`crates/lucene-search/tests/geo_points_fixtures.rs` -- ~1100 queries over a
four-segment index with deletions, plus 80 over a 24 000-point segment
(distance-feature narrowing, the sort comparator's sampled updates), each
answered identically (hits, score/sort-value/distance bits) over Lucene's index
**and** over this port's index of the same documents, whose packed points equal
Lucene's byte for byte. Write path: `scripts/verify-write-path.sh` runs
`write_geo_points_fixture` -> `VerifyGeoPoints` (`CheckIndex`, then Lucene
replays every query over the Rust-written index).

Bench: `scripts/bench-micro.sh --bench geo_points` (`GeoPointsMicro.java` /
`micro_geo_points.rs`, one million points, counting collector, cross-engine
digests, query set `geo-queries-v2.tsv`; 2026-10-02 T9.2 review, noise 1.22x):
`geo_box` 1.53x, `geo_distance` 3.36x, `geo_polygon` 3.45x,
`geo_distance_sort` 0.98x and `geo_nearest` 1.18x (both in noise),
`geo_distance_feature` 1.13x (in noise), `geo_polygon_within` 2.94x,
`geo_polygon_disjoint` 3.22x, `geo_points_contains` 2.20x, `geo_line` 2.37x,
`geo_circle_within` 3.95x, `geo_xy_box` 1.70x, `geo_xy_distance` 1.42x,
`geo_xy_polygon` 1.64x. `nearest` navigates lazily and the distance sort reads
the resolved single-valued array (both needed to reach 1.0).

### [util-geo.md](../parity/util-geo.md): lucene-util -- geo (M9 T9.1)

`org.apache.lucene.geo`: pure geometry, no index format
of its own -- but the encodings and the Tessellator's triangles *are* what
`LatLonPoint`/`LatLonShape` index ([geo-points.md](../parity/geo-points.md),
[geo-shapes.md](../parity/geo-shapes.md)), so everything here is held to Java's bits.
Tests: `fixtures/src/GenGeo.java`, `GenGeoTessellator.java`,
`GenGeoParsers.java` (corpus: `GeoCorpus.java`) ->
`crates/lucene-util/tests/geo_fixtures.rs`. Bench: `scripts/bench-micro.sh
--bench geo` (`GeoMicro.java` / `micro_geo.rs`, cross-engine digests;
2026-10-02, noise 1.07x): `tessellate_latlon` 1.36x,
`tessellate_latlon_checked` 1.33x, `tessellate_xy_checked` 1.04x (in noise),
`component_build` 3.54x, `component_relate` 1.44x, `component_contains`
1.50x, `component_intersects_triangle` 1.28x, `haversin_meters` 1.01x.

### [plugin-geo.md](../parity/plugin-geo.md): how the geo queries reach the native engine

What OpenSearch 3.8.0 builds for `geo_bounding_box`, `geo_distance`,
`geo_polygon` and `geo_shape` (on `geo_point` and `geo_shape` fields) and
for the `_geo_distance` sort, and how each reaches the native engine. The
Java query objects are read by class (the four geo query classes are
package-private: their constructor arguments through the plugin's `Reflect`)
in `opensearch-plugin/.../GeoEncoder.java`, sent as the query tree's nodes
15-19, decoded in `lucene-ffi/src/jvm_reader.rs` and run as
`DocumentClause`s -- the T9.2/T9.3 ports as leaves of the scorer tree, their
sets before deletions, keyed for the query cache by the node's bytes (Java's
`equals`: every constructor argument). Proved against a stock node by
`opensearch-plugin/e2e/geo_matrix.py` (in `scripts/verify-opensearch.sh`) and
against Lucene in process by `NativeSelfTest.geo`.
