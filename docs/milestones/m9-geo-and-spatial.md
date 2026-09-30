# M9 — Geo and spatial

> **Goal:** points and shapes on a sphere and a plane -- indexing, the
> queries, distance sorting and doc-values forms -- as Lucene has them, so
> OpenSearch's `geo_point` and `geo_shape` run natively.

| | |
|---|---|
| **Effort** | L |
| **Depends on** | [M7](m7-core-complete.md) (`document` fields, multi-dimension BKD writing) |
| **Unblocks** | native `geo_distance`, `geo_bounding_box`, `geo_shape`, geo sorting |
| **Status** | not started |

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
- **T9.2** — `LatLonPoint`/`XYPoint`: fields, queries, distance sort.
- **T9.3** — Shapes: `LatLonShape`/`XYShape` writing and querying, including
  the doc-values shape encoding.
- **T9.4** — `spatial3d`.
- **T9.5** — `spatial-extras` (prefix trees, `SpatialStrategy`s).
- **T9.6** — Plugin wiring for OpenSearch's geo queries and sort.

## Acceptance criteria

- [ ] Every geo query returns the same hits as Lucene on a generated corpus
      of random points and shapes, including the antimeridian, the poles and
      degenerate polygons.
- [ ] The `Tessellator` produces Lucene's triangles, or fails where Lucene
      fails, on a corpus of real-world polygons.
- [ ] Real Lucene reads Rust-written point and shape indices.
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
