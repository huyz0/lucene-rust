# OpenSearch plugin -- geo queries, sorting (M9 T9.6)

[Index](../parity.md). Rows: OpenSearch query shapes, not ports; **native** (engine) or **falls back** (named reason, to Lucene).

OpenSearch 3.8.0's `geo_bounding_box`, `geo_distance`, `geo_polygon`,
`geo_shape` and `_geo_distance` sort: read by `opensearch-plugin/.../GeoEncoder.java`,
decoded by `lucene-ffi/src/jvm_reader.rs`, run as `DocumentClause`s, cached by
node bytes. Tests: `opensearch-plugin/e2e/geo_matrix.py` and
`NativeSelfTest.geo` ([M9 evidence](../milestones/m9-geo-and-spatial.md#ledger-evidence)).

| Java | Rust | Status |
|---|---|---|
| `document/LatLonPoint.newBoxQuery` (its two-dimension `PointRangeQuery`; a `ConstantScoreQuery` of two across the dateline) -- OpenSearch's `geo_bounding_box` and `geo_shape` envelope on a `geo_point`, inside `IndexOrDocValuesQuery` with `LatLonDocValuesField.newSlowBoxQuery` | `lucene-ffi/src/jvm_reader.rs::decode_node` (node 15), `lucene-search/src/extended_query.rs::PointRangeQuery` | **native.** The index side of `IndexOrDocValuesQuery` runs (same matches); only `LatLonPoint`'s range; other multi-dim/4-byte ranges fall back (`points_width`). |
| `document/LatLonPointDistanceQuery` -- `geo_distance`, and a `geo_shape` circle, on a `geo_point` | `lucene-ffi/src/jvm_reader.rs::decode_geo` (node 16), `lucene-search/src/extended_query.rs::DocumentClause` | **native.** |
| `document/LatLonPointQuery` -- `geo_polygon`, `geo_shape` polygons and multipolygons on a `geo_point` (OpenSearch asks `INTERSECTS` only; every relation is encoded) | `lucene-ffi/src/jvm_reader.rs::{decode_geo, decode_geometries}` (node 17) | **native**, every `LatLonGeometry` (`Point`, `Line`, holed `Polygon`, `Rectangle`, `Circle`). |
| `document/LatLonShapeQuery`, `document/LatLonShapeBoundingBoxQuery` -- every query on a `geo_shape` field (`LatLonShape.newGeometryQuery`, incl. `geo_bounding_box` and `geo_distance` on one; a `CONTAINS` of several geometries or across the dateline is a `BooleanQuery` of them, encoded as one) | `lucene-ffi/src/jvm_reader.rs::decode_geo` (nodes 18, 19) | **native**, all four relations, inline or `indexed_shape` (prefetched). |
| `document/LatLonPointSortField`, `document/LatLonPointDistanceComparator` -- OpenSearch's `_geo_distance` with one origin, metres, ascending, `min` | `lucene-ffi/src/jvm_reader.rs::decode_geo_sort` (comparator 0), `lucene-search/src/top_field.rs::SortField::with_source` | **native**: sort key as value (JVM gets its `haversin2`), Lucene's `compareBottom` (`LeafFieldComparator::{set_bottom, compare_bottom}`). `search_after` falls back (`search_after_geo`: metres don't map to keys). At 10,000+ docs, points come from the decoded segment (`SortColumn::Multi`). |
| OpenSearch's `GeoDistanceSortBuilder` comparator source (several origins, a unit, descending, `max`/`avg`/`median`) -- OpenSearch, not Lucene | `lucene-search/src/document/geo/sort.rs::{OpenSearchGeoDistanceSort, DistanceMode}`, `lucene-ffi/src/jvm_reader.rs::decode_geo_sort` (comparator 1) | **native** for `distance_type: arc` (`SloppyMath.haversinMeters`, `DistanceUnit.convert`, `SortingNumericDoubleValues`' sort, `MultiValueMode`'s pick, missing `+Infinity`), plus `search_after`; `plane` falls back (`sort_geo_plane`: `Math.cos`), as does nested (`sort_geo_nested`). |
| OpenSearch's legacy `geo_shape` mapping (`tree: quadtree/geohash`, spatial-extras' `RecursivePrefixTreeStrategy`) | -- | **falls back** (`query_<Class>`): deprecated (OpenSearch warns on it); spatial-extras ported (T9.5), not wired in. |
