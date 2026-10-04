# lucene-util / lucene-search -- spatial-extras (M9 T9.5)

[Index](../parity.md). **Ported**: every class of the jar
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

| Java | Rust | Status |
|---|---|---|
| Spatial4j `context/SpatialContext`, `context/SpatialContextFactory`, Lucene `spatial4j/Geo3dSpatialContextFactory` | `lucene-util/src/spatial4j/context.rs::{SpatialContext, SpatialContextFactory, FactoryKind}` | **ported**: `makeSpatialContext(args)` (`geo`, `distCalculator` incl. `geo3d`, `worldBounds` as WKT, `normWrapLongitude`, `planetModel`, class-name settings), Java's validation messages, `toString` (`SpatialContext.GEO` for `geo_context()`). Differs: factory classes are a `FactoryKind`, not reflection. |
| Spatial4j `shape/Shape`, `Point`, `Rectangle`, `Circle`, `SpatialRelation`, `BaseShape`, `impl/PointImpl`, `impl/RectangleImpl`, `impl/CircleImpl`, `impl/GeoCircle`, `ShapeCollection`, `impl/BBoxCalculator`, `impl/InfBufLine`, `impl/BufferedLine`, `impl/BufferedLineString` | `lucene-util/src/spatial4j/shape.rs`, `lucene-util/src/spatial4j/point.rs`, `lucene-util/src/spatial4j/rectangle.rs`, `lucene-util/src/spatial4j/circle.rs`, `lucene-util/src/spatial4j/collection.rs`, `lucene-util/src/spatial4j/bbox_calculator.rs`, `lucene-util/src/spatial4j/buffered_line.rs` | **ported**, bit-exact: traits over `Arc<dyn ...>`, `instanceof` via `as_point`/`as_rectangle`/`as_circle`; `GeoCircle` is a `CircleImpl` with its inverse circle. Differs: no `reset`/reuse arguments; `RectangleImpl.equals` compares any rectangle's bounds (Java: `ClassCastException` vs a Geo3D one); no `hashCode`. `GeoCircle.toString`'s `%.1f`/`%.2f` round as Java's `Formatter`. |
| Spatial4j `distance/DistanceUtils`, `DistanceCalculator`, `AbstractDistanceCalculator`, `GeodesicSphereDistCalc` (+ `Haversine`, `LawOfCosines`, `Vincenty`), `CartesianDistCalc` | `lucene-util/src/spatial4j/distance.rs` | **ported**, bit-exact (`toRadians` = `degrees * DEGREES_TO_RADIANS`, Spatial4j's own). Not ported: deprecated `vector*` helpers. |
| Spatial4j `shape/ShapeFactory` (+ builders), `impl/ShapeFactoryImpl` | `lucene-util/src/spatial4j/shape_factory.rs` | **ported**: builders are traits; factory methods take the context. |
| Spatial4j `io/WKTReader`, `io/BinaryCodec`, `io/GeohashUtils` | `lucene-util/src/spatial4j/wkt.rs`, `lucene-util/src/spatial4j/binary_codec.rs`, `lucene-util/src/spatial4j/geohash.rs` | **ported**, codec byte-identical; WKT errors with Java's class/message/offset. Differs: `decodeBoundary` fails on a non-geohash char (Java indexes unchecked); nesting deeper than 64 is a `StackOverflowError` (WKT: `ParseException`) rather than a real stack overflow. Not ported (unused): GeoJSON, Polyshape, legacy readers/writers, `WKTWriter`, `SupportedFormats`, Jackson, `Range`. |
| s2-geometry `S2CellId`, `S2Cell` (vertices), `S2LatLng.toPoint`, `S2Point`, `S2Projections` (quadratic), `S2.Metric` | `lucene-util/src/s2/mod.rs` | **ported**, bit-exact for what `S2PrefixTree` and `Geo3dShapeFactory.getS2CellShape` use (ids, parents/children, levels, faces, tokens, containment, unsigned order, vertices, `MAX_WIDTH`). Not ported (unused): regions, coverers, loops, polygons. |
| Lucene `spatial4j/Geo3dShape`, `Geo3dPointShape`, `Geo3dRectangleShape`, `Geo3dCircleShape`, `Geo3dShapeFactory`, `Geo3dBinaryCodec`, `Geo3dDistanceCalculator` | `lucene-util/src/spatial_extras/spatial4j.rs` | **ported**, bit-exact; one struct with a kind for the four shapes. Differs: `equals` compares serialized forms; `toString` is `Geo3D:` + class name. Kept Lucene quirk: relating a Geo3D shape to a plain Spatial4j rectangle reads `GeoArea.getRelationship` inverted (containing the box answers `WITHIN`). |
| `spatial/prefix/tree/SpatialPrefixTree`, `Cell`, `CellCanPrune`, `CellIterator`, `FilterCellIterator`, `SingletonCellIterator`, `TreeCellIterator`, `SpatialPrefixTreeFactory`, `S2ShapeFactory` | `lucene-util/src/spatial_extras/prefix_tree/mod.rs` | **ported**, byte-identical tokens. Differs: cells/iterators are traits handing out owned cells (iterators keep a copy for `thisCell()`; `next_detached` skips it); `readCell(term, scratch)` is `read_cell_into`; no `getTokenBytes*(BytesRef)` scratch args; `makeSPT` names the four trees only (others: `ClassNotFoundException`). |
| `spatial/prefix/tree/LegacyPrefixTree`, `LegacyCell`, `QuadPrefixTree`, `GeohashPrefixTree`, `PackedQuadPrefixTree` | `lucene-util/src/spatial_extras/prefix_tree/legacy.rs`, `lucene-util/src/spatial_extras/prefix_tree/quad.rs`, `lucene-util/src/spatial_extras/prefix_tree/geohash.rs`, `lucene-util/src/spatial_extras/prefix_tree/packed_quad.rs` | **ported**, byte-identical: tokens, levels, shapes, relations, children, `readCell`, iterators, level-for-distance, pruning. A quad term deeper than the tree is Java's `ArrayIndexOutOfBoundsException` from `getShape()`. Kept Java quirks: packed-quad `isPrefixOf` compares only its own level's bits; `readCell` of a term < 8 bytes throws. Not ported (unused): `QuadPrefixTree.buildNotRobustly`/`checkBattenbergNotRobustly`/`printInfo`. |
| `spatial/prefix/tree/S2PrefixTree`, `S2PrefixTreeCell` | `lucene-util/src/spatial_extras/prefix_tree/s2.rs` | **ported**, byte-identical at arity 1-3. `readCell` of an empty term throws as Java. Differs: a term past 30 levels has an empty token (Java: `ArrayIndexOutOfBoundsException`); `isPrefixOf`/`compareToNoLeaf` against the world cell answer `false`/`1` (Java: `NullPointerException`). |
| `spatial/prefix/tree/NumberRangePrefixTree` (+ `UnitNRShape`, `SpanUnitsNRShape`, `NRShape`, `NRCell`), `DateRangePrefixTree` | `lucene-util/src/spatial_extras/prefix_tree/number_range.rs`, `lucene-util/src/spatial_extras/prefix_tree/date_range.rs` | **ported**, byte-identical terms; parse/format round trips on the hybrid calendar and `JAVA_UTIL_TIME_COMPAT_CAL`, sub-cell counts, relations, `compareTo`, `roundToLevel`. `NRCell` reuse redesigned as value `UnitNRShape`s. Last-level children throw Java's message. Differs: number-range vs geometric relate is an error on the second bounce (Java recurses to `StackOverflowError`); corrupt date levels use Java's wrapping `int` arithmetic. |
| `java.util.GregorianCalendar` (subset `DateRangePrefixTree` uses: UTC, lenient, `ERA`..`MILLISECOND`, the Julian/Gregorian cutover, `getActualMinimum`/`getActualMaximum`) | `lucene-util/src/spatial_extras/prefix_tree/java_calendar.rs` | **ported** from OpenJDK 21 (`GregorianCalendar`/`BaseCalendar`/`JulianCalendar`, cutover rules `getFixedDateMonth1`, `actualMonthLength`). Not ported (unused): time zones, locales, week fields, `roll`/`add`. |
| `spatial/query/SpatialOperation`, `SpatialArgs`, `SpatialArgsParser` | `lucene-util/src/spatial_extras/query.rs` | **ported**: names/aliases, `evaluate`, `calcDistanceFromErrPct`, `resolveDistErr`, `validate`, `toString`, parser results and errors on 88 strings. `UnsupportedSpatialOperation` is `SpatialOperation::unsupported`. |
| (prefix-tree unit tests) | `lucene-util/src/spatial_extras/prefix_tree/tests.rs` | **rust-only**: construction errors, iterator protocol, cutover calendar edges. |
| (module roots and unit tests) | `lucene-util/src/spatial4j/mod.rs` (`Error`, `Double.compare`/`Double.toString`/`%.Nf` helpers), `lucene-util/src/spatial_extras/mod.rs`, `lucene-util/src/spatial4j/tests.rs`, `lucene-util/src/s2/tests.rs`, `lucene-util/src/spatial_extras/spatial4j_tests.rs` | **rust-only**: module roots and unit tests. |
| `spatial/SpatialStrategy` (+ `makeRecipDistanceValueSource`) | `lucene-search/src/spatial/mod.rs` | **ported**: a trait over `Arc<dyn Shape>`; fields are `IndexableField`s, queries `DocumentQuery`s, value sources `DoubleValuesSource`s; `as_prefix_tree` is Java's `(PrefixTreeStrategy)` cast. |
| `spatial/prefix/PrefixTreeStrategy`, `RecursivePrefixTreeStrategy`, `TermQueryPrefixTreeStrategy`, `NumberRangePrefixTreeStrategy`, `CellToBytesRefIterator`, `BytesRefIteratorTokenStream` | `lucene-search/src/spatial/prefix/mod.rs` | **ported**, byte-identical fields (pre-analyzed token stream). Kept Java quirk: a points-only RPT indexing a non-point while pruning does not refuse it. `NumberRangePrefixTreeStrategy` is an RPT with a flag; the TermQuery strategy's `TermInSetQuery` is a constant-score union. |
| `spatial/prefix/AbstractPrefixTreeQuery`, `AbstractVisitingPrefixTreeQuery`, `IntersectsPrefixTreeQuery`, `WithinPrefixTreeQuery`, `ContainsPrefixTreeQuery` (+ RPT's point `TermQuery`) | `lucene-search/src/spatial/prefix/query.rs` | **ported**, same hits: `VisitorTemplate` step for step (seekCeil leap-frogging, scan level); `VNode` a stack of owned nodes; `SmallDocSet` a sorted `Vec`; quad trees take `visit_quad` (value cells). RPT's points-only point query is a `TermQuery` scored bit for bit (BM25, stats summed over segments). |
| `spatial/prefix/PrefixTreeFacetCounter`, `HeatmapFacetCounter`, `NumberRangePrefixTreeStrategy.Facets` | `lucene-search/src/spatial/prefix/facets.rs` | **ported**, same counts (heatmaps incl. dateline wrap, date facets). Java's `ClassCastException` for non-rectangle (S2) or non-unit cells; world `UnitNRShape.clone()` throws as Java. `topAcceptDocs` is a global `FixedBitSet`. |
| `spatial/bbox/BBoxStrategy`, `BBoxValueSource`, `BBoxSimilarityValueSource`, `BBoxOverlapRatioValueSource` | `lucene-search/src/spatial/bbox.rs` | **ported**, same hits and bit-identical scores/explanations (incl. Java's reuse of `targetRatio` in a detail). The box source builds an unnormalised `RectangleImpl` (Java's `reset`). |
| `spatial/vector/PointVectorStrategy`, `DistanceValueSource` | `lucene-search/src/spatial/vector.rs` | **ported**: boxes (two SHOULD ranges across the dateline), circles (`DistanceRangeQuery`); distances bit for bit. |
| `spatial/serialized/SerializedDVStrategy` (+ `PredicateValueSourceQuery`, `ShapeDocValueSource`) | `lucene-search/src/spatial/serialized.rs` | **ported**: codec bytes identical (Spatial4j and Geo3D), each live doc verified. Differs: `indexLastBufSize` (a buffer heuristic) dropped. |
| `spatial/composite/CompositeSpatialStrategy`, `CompositeVerifyQuery`, `IntersectsRPTVerifyQuery` | `lucene-search/src/spatial/composite.rs` | **ported**: optimized intersects and the general verify path. |
| `spatial/ShapeValues`, `ShapeValuesSource`, `spatial/util/ShapeValuesPredicate`, `ShapeAreaValueSource`, `DistanceToShapeValueSource`, `ReciprocalDoubleValuesSource`, `CachingDoubleValueSource`, `ShapeFieldCache`, `ShapeFieldCacheProvider`, `ShapeFieldCacheDistanceValueSource`, `spatial/prefix/PointPrefixTreeFieldCacheProvider` | `lucene-search/src/spatial/util.rs` | **ported**, values bit for bit. Differs: Java's per-reader `WeakHashMap` cache is recomputed per `get_values` call. |
| (the boolean matcher, unit tests) | `lucene-search/src/spatial/bool_query.rs`, `lucene-search/src/spatial/tests.rs` | **rust-only**: the `BooleanQuery` BBox/point-vector build (MUST, SHOULD + `minimumNumberShouldMatch`, MUST_NOT); unit tests. |
