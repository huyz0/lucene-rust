# lucene-index / lucene-search -- geo points (M9 T9.2)

[Index](../parity.md). The point half of Lucene's geo `document` classes: the
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

| Java | Rust | Status |
|---|---|---|
| `document/LatLonPoint` (fields, `newBoxQuery`, `newDistanceQuery`, `newPolygonQuery`, `newGeometryQuery`, `newDistanceFeatureQuery`, `nearest`) | `lucene-index/src/document/geo.rs::LatLonPoint`, `lucene-search/src/document/geo/mod.rs::lat_lon_point` (edge tests: `lucene-search/src/document/geo/tests.rs`) | **ported**: two 4-byte dims (`encodeLatitude`/`encodeLongitude` + `intToSortableBytes`; `encodeCeil` for a box's lower corner). `newBoxQuery`'s special cases (`minLat == 90`, `minLon == maxLon == 180` match nothing; `minLon == 180` wraps; a dateline box is two `PointRangeQuery`s in a constant-score disjunction). `newGeometryQuery`: one rectangle/circle becomes a box/distance query, `CONTAINS` a conjunction of `LatLonPointQuery`s per point. Bitset-filling visitors fail with `Corrupted` on a doc id outside the segment (as Java's `FixedBitSet` throws). |
| `document/LatLonPointDistanceQuery` | `lucene-search/src/document/geo/point_queries.rs::LatLonPointDistanceQuery` | **ported**: encoded bounding box (two longitude ranges across the dateline), `GeoUtils.relate` with the radius' sort key and `axisLat`, `DistancePredicate` per point, the inverse walk when every document has one point and cost > half the segment. |
| `document/LatLonPointQuery` (+ the point half of `SpatialQuery`) | `lucene-search/src/document/geo/point_queries.rs::{LatLonPointQuery, SpatialVisitor}` | **ported**, all four relations, `SpatialQuery`'s scorer choice (nothing / everything / `hasAnyHits` first for `WITHIN`/`DISJOINT` multi-valued / sparse, intersects-dense, inverse-dense, dense, contains-dense visitors). `QueryRelation` (`ShapeField.QueryRelation`) lives in `geo/mod.rs`; the shape half is in [geo-shapes.md](geo-shapes.md). Differs: a field not two indexed 4-byte dims is an error (`check_points_shape`, also in the feature query and `nearest`) where Java would fail decoding. |
| `document/XYPointField`, `document/XYPointInGeometryQuery` | `lucene-index/src/document/geo.rs::XYPointField`, `lucene-search/src/document/geo/point_queries.rs::XYPointInGeometryQuery`, `lucene-search/src/document/geo/mod.rs::xy_point_field` | **ported**: `XYEncodingUtils` per dimension; every factory is an `XYPointInGeometryQuery`, `checkCompatible`'s messages. |
| `document/LatLonDocValuesField`, `document/XYDocValuesField` | `lucene-index/src/document/geo.rs::{LatLonDocValuesField, XYDocValuesField}`, `lucene-search/src/document/geo/mod.rs::{lat_lon_doc_values_field, xy_doc_values_field}` | **ported**: one `SORTED_NUMERIC` long per point (latitude high, longitude low); `LatLonPoint`'s factory special cases. |
| `document/LatLonDocValuesBoxQuery`, `document/LatLonDocValuesQuery`, `document/XYDocValuesPointInGeometryQuery` | `lucene-search/src/document/geo/doc_values.rs::{LatLonDocValuesBoxQuery, LatLonDocValuesQuery, XYDocValuesPointInGeometryQuery}` | **ported**: two-phase `matches()` (any/all/none/`withinPoint` per relation); a field of another doc-values type matches nothing. |
| `document/LatLonPointDistanceFeatureQuery` | `lucene-search/src/document/geo/distance_feature.rs::LatLonPointDistanceFeatureQuery` | **ported**: `weight * pivot / (pivot + haversinMeters)` of the closest value; `setMinCompetitiveScore` narrowing as Java (`Math.nextUp` of the worst score, binary-searched max distance, box documents swapped in under an eighth of the lead cost, every 32nd update past the 256th); the narrowed cost is `DocIdSetBuilder`'s (via `IntersectVisitor::grow`), so lower-bound totals equal Lucene's; a threshold is pushed only above the leaf's last push, so a NaN pivot is never pruned. Tests: also `GenDistanceFeaturePruning` (totals, relation, hits, NaN pivots). |
| `document/LatLonPointSortField`, `document/LatLonPointDistanceComparator`, `document/XYPointSortField`, `document/XYPointDistanceComparator` | `lucene-search/src/document/geo/sort.rs::{LatLonPointSortField, LatLonDistance, XYPointSortField, XYDistance}` | **ported**: `TopFieldCollector` with the comparator (`compareBottom`'s bounding-box rejection rebuilt per `setBottom` for 1024 calls then every 64th; `XY` only below `Float.MAX_VALUE`), values `haversin2` of the min sort key, missing last at infinity, ties by doc. Also a `FieldComparatorSource` (`comparator_source`) for `top_field::search_sorted`. `setMissingValue` only `+Infinity`; `n == 0` refused with Java's `numHits must be > 0` before the field is read. |
| `document/NearestNeighbor` | `lucene-search/src/document/geo/nearest.rs::{nearest, NearestHit, NearestHits}` | **ported**: best-first by `approxBestDistance` in a port of `java.util.PriorityQueue` (Java's tie order), `maybeUpdateBBox` sampled past 1024, deletions skipped, one hit per matching point; `total_hits` = the field's doc count. Navigates `lucene_codecs::points::PointTreeNode`. Differs: a field not two 4-byte dims, or a leaf doc outside the segment (corrupt `.kdd`), is an error. |
