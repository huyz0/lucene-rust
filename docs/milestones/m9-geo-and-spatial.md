# M9 — Geo and spatial

> **Goal:** points and shapes on a sphere and a plane -- indexing, the
> queries, distance sorting and doc-values forms -- as Lucene has them, so
> OpenSearch's `geo_point` and `geo_shape` run natively.

| | |
|---|---|
| **Effort** | L |
| **Depends on** | [M7](m7-core-complete.md) (`document` fields, multi-dimension BKD writing) |
| **Unblocks** | native `geo_distance`, `geo_bounding_box`, `geo_shape`, geo sorting |
| **Status** | in progress: T9.1, T9.2, T9.3 done (2026-10-02) |

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
  --bench geo`; ratios in `docs/parity.md`'s geo section.
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
  (`c1-lazy-blocktree.md` F-9), and per-cell allocation. The review closed
  four ways corrupt bytes could panic or abort: quad/packed-quad terms
  deeper than the tree, a date term's year overflow, an S2 term with no
  level, and unbounded nesting in binary shapes, geo3d streams and WKT.
- **T9.6** — Plugin wiring for OpenSearch's geo queries and sort.

## Acceptance criteria

- [ ] Every geo query returns the same hits as Lucene on a generated corpus
      of random points and shapes, including the antimeridian, the poles and
      degenerate polygons.
- [ ] The `Tessellator` produces Lucene's triangles, or fails where Lucene
      fails, on a corpus of real-world polygons.
- [x] Real Lucene reads Rust-written point and shape indices.
- [ ] Native `geo_distance` and `geo_shape` searches agree with a stock node
      in the plugin's matrix, and are no slower than Lucene.

## Risks and unknowns

- **Floating-point geometry.** Encoding and relation tests are
  quantization-sensitive; the port must follow Lucene's arithmetic exactly,
  or edge points flip in and out of results.

## Exit artifacts

- Geo fixture generators and the real-world polygon corpus
- `docs/parity.md` rows for `geo`, the geo `document` classes, `spatial3d`,
  `spatial-extras`
