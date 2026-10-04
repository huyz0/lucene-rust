# OpenSearch plugin -- geo queries and geo sorting (M9 T9.6)

[Index](../parity.md). Rows here are OpenSearch query shapes rather than ports: the status is **native** (served by the engine) or **falls back** (a named reason, Lucene serves it).

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

| Java | Rust | Status |
|---|---|---|
| `document/LatLonPoint.newBoxQuery` (its two-dimension `PointRangeQuery`; a `ConstantScoreQuery` of two across the dateline) -- OpenSearch's `geo_bounding_box` and `geo_shape` envelope on a `geo_point`, inside `IndexOrDocValuesQuery` with `LatLonDocValuesField.newSlowBoxQuery` | `lucene-ffi/src/jvm_reader.rs::decode_node` (node 15), `lucene-search/src/extended_query.rs::PointRangeQuery` | **native.** The index side of the `IndexOrDocValuesQuery` runs (both sides match the same documents); only `LatLonPoint`'s own range is taken, any other multi-dimension or 4-byte range falls back (`points_width`). |
| `document/LatLonPointDistanceQuery` -- `geo_distance`, and a `geo_shape` circle, on a `geo_point` | `lucene-ffi/src/jvm_reader.rs::decode_geo` (node 16), `lucene-search/src/extended_query.rs::DocumentClause` | **native.** |
| `document/LatLonPointQuery` -- `geo_polygon`, `geo_shape` polygons and multipolygons on a `geo_point` (OpenSearch asks `INTERSECTS` only; every relation is encoded) | `lucene-ffi/src/jvm_reader.rs::{decode_geo, decode_geometries}` (node 17) | **native**, every `LatLonGeometry` (`Point`, `Line`, `Polygon` with holes, `Rectangle`, `Circle`). |
| `document/LatLonShapeQuery`, `document/LatLonShapeBoundingBoxQuery` -- every query on a `geo_shape` field (`LatLonShape.newGeometryQuery`, incl. `geo_bounding_box` and `geo_distance` on one; a `CONTAINS` of several geometries or across the dateline is a `BooleanQuery` of them, encoded as one) | `lucene-ffi/src/jvm_reader.rs::decode_geo` (nodes 18, 19) | **native**, all four relations, inline and `indexed_shape` geometries alike (OpenSearch fetches the indexed one before the query is built). |
| `document/LatLonPointSortField`, `document/LatLonPointDistanceComparator` -- OpenSearch's `_geo_distance` with one origin, metres, ascending, `min` | `lucene-ffi/src/jvm_reader.rs::decode_geo_sort` (comparator 0), `lucene-search/src/top_field.rs::SortField::with_source` | **native**, a hit's value the sort key (the JVM reports `haversin2` of it, as the comparator's `value`), with Lucene's `compareBottom` (a value outside the bottom's bounding box is not measured; `LeafFieldComparator::{set_bottom, compare_bottom}`). `search_after` stays Lucene's (`search_after_geo`): `compareTop` compares metres, which do not map back to keys exactly. Stage 3: in a segment of 10,000 documents or more the points are read from the segment's decoded copy from the sort's second use (`SortColumn::Multi`, as numeric sort keys' columns) -- the same values, decoded once. |
| OpenSearch's `GeoDistanceSortBuilder` comparator source (several origins, a unit, descending, `max`/`avg`/`median`) -- OpenSearch, not Lucene | `lucene-search/src/document/geo/sort.rs::{OpenSearchGeoDistanceSort, DistanceMode}`, `lucene-ffi/src/jvm_reader.rs::decode_geo_sort` (comparator 1) | **native** for `distance_type: arc` (`SloppyMath.haversinMeters`, `DistanceUnit.convert`, `SortingNumericDoubleValues`' sort, `MultiValueMode`'s pick, missing at `+Infinity`), `search_after` included; `plane` falls back (`sort_geo_plane`: `Math.cos`), and so does a nested one (`sort_geo_nested`). |
| OpenSearch's legacy `geo_shape` mapping (`tree: quadtree/geohash`, spatial-extras' `RecursivePrefixTreeStrategy`) | -- | **falls back** (`query_<Class>`): a deprecated mapping (OpenSearch logs a deprecation for its parameters); spatial-extras itself is ported (T9.5), but not wired to the plugin. |
