# lucene-index / lucene-search -- geo shapes (M9 T9.3)

[Index](../parity.md). The shape half of Lucene's geo `document` classes:
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

| Java | Rust | Status |
|---|---|---|
| `document/ShapeField` (`TYPE`, `Triangle`, `encodeTriangle`, `decodeTriangle`, `resolveTriangleType`, `DecodedTriangle`, `QueryRelation`) | `lucene-index/src/document/shape.rs::{ShapeField, ShapeTriangle, DecodedTriangle, TriangleType}`, `lucene-search/src/document/geo/mod.rs::QueryRelation` (unit tests: `lucene-index/src/document/shape/tests.rs`) | **ported**, byte-identical: seven 4-byte dims, four indexed; westmost-vertex rotation, CCW turn by `GeoUtils.orient`, eight layouts, edge bits. Differs: `decodeTriangle` returns the value; `Could not encode the provided triangle` is an error (unreached: 1500 fixture + 20M random near-collinear triangles). |
| `document/LatLonShape`, `document/XYShape` (field factories) | `lucene-index/src/document/shape.rs::{LatLonShape, XYShape}` | **ported**: polygons through the `Tessellator` (with/without `checkSelfIntersections`), lines as `(a, b, a)`, points as `(p, p, p)`; one function per Java overload. Java's on-disk quirk kept: a polygon triangle's second edge flag comes from its first (`isEdgefromPolygon(0)` twice). `createDocValueField(name, Field[])` also offered for `XYShape`. |
| `document/ShapeDocValues` (with `TreeNode`, `Writer`, `Reader`, `ShapeComparator`), `document/LatLonShapeDocValues`, `document/XYShapeDocValues` | `lucene-index/src/document/shape_doc_values.rs::{ShapeDocValues, ShapeEncoding}`, `lucene-index/src/document/shape.rs::{LatLonShapeDocValues, XYShapeDocValues}` (unit tests: `lucene-index/src/document/shape_doc_values/tests.rs`) | **ported**, byte-identical: median-split tree (`IntroSelector` swap for swap), pulled-up bounds, subtree sizes, centroid (`Math.hypot` via `strict_math::hypot`), version-0 serialization, `relate` with pruning. Differs: corrupt values, a skip outside the value, and a tree deeper than 128 are errors (Java throws or overflows its stack); the query opens a lat/lon value without building centroid/box objects. |
| `document/ShapeDocValuesField`, `document/LatLonShapeDocValuesField`, `document/XYShapeDocValuesField` | `lucene-index/src/document/shape.rs::{LatLonShapeDocValuesField, XYShapeDocValuesField}` | **ported**: one `BINARY` value per doc, norms omitted; `numberOfTerms`, centroid, bounding box, highest dimension. Not ported: `ShapeDocValuesField.newGeometryQuery` (Java always throws "not yet supported"). |
| `document/SpatialQuery` (shape half: `EncodedRectangle`, relation scorers over triangles) | `lucene-search/src/document/geo/shape_queries.rs::EncodedRectangle`, `lucene-search/src/document/geo/point_queries.rs::{SpatialVisitor, spatial_score_leaf}` | **ported** (point half: [geo-points.md](geo-points.md)). `SpatialVisitor` has `within` (`containsTriangle`) and the up-front `contains()` refusal; a run of documents in an inside cell is taken at once. Same scorer family as points. |
| `document/LatLonShapeQuery`, `document/XYShapeQuery` | `lucene-search/src/document/geo/shape_queries.rs::{LatLonShapeQuery, XYShapeQuery}`, `lucene-search/src/document/geo/mod.rs::{lat_lon_shape, xy_shape}` (edge tests: `lucene-search/src/document/geo/shape_queries/tests.rs`) | **ported**: cells related by the `Component2D`; each triangle tested as the point/line/triangle it is. Java's factory special cases: one rectangle is a box query; `CONTAINS` of several geometries a constant-score conjunction; `WITHIN` a line refused for lat/lon only. |
| `document/LatLonShapeBoundingBoxQuery` | `lucene-search/src/document/geo/shape_queries.rs::LatLonShapeBoundingBoxQuery` | **ported**: encoded box (ceil/floor, `validateMinLon`), dateline halves, `intersectBBoxWithRangeBBox`/`compareBBoxToRangeBBox` on `int`s in Java's unsigned order, `contains()`'s refusal of a dateline box. `newBoxQuery` splits a `CONTAINS` dateline box into two `MUST` clauses (`MustConjunction`, scoring the sum -- twice the boost); that score is unreachable in practice (Lucene 10.5.0 also matches nothing there) and only unit-tested. |
| `document/BaseShapeDocValuesQuery`, `document/LatLonShapeDocValuesQuery`, `document/XYShapeDocValuesQuery` | `lucene-search/src/document/geo/shape_queries.rs::{LatLonShapeDocValuesQuery, XYShapeDocValuesQuery}` | **ported**: every value related to the `Component2D` (`WITHIN` needs the whole box inside, `DISJOINT` negates), `CONTAINS` refused, non-`BINARY` field has no values. Constructors take any geometries (package-private in Java; fixture via `ShapeAccess`). `newSlowDocValuesBoxQuery` turns a `CONTAINS` dateline box into the indexed conjunction, as Java. |
