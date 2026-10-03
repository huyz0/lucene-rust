"""The geo half of verify_opensearch.py's matrix (M9 T9.6): geo_point and
geo_shape indices whose points and shapes reach the poles and the
antimeridian, and every geo query and _geo_distance sort shape OpenSearch
builds on them -- each answered by the plugin and by stock Lucene, which must
agree hit for hit, score for score and sort value for sort value.

Rows are (name, body, expect) as verify_opensearch.matrix() has them:
"native" must run natively on every shard; anything else is the fallback
reason expected (sort shapes OpenSearch answers with its own comparator, in
the sort commit's rows).
"""
import json
import random

WORDS = "alpha beta gamma delta epsilon zeta eta theta".split()


def mapping():
    return {"properties": {
        "body": {"type": "text"},
        "tag": {"type": "keyword"},
        "n": {"type": "long"},
        "loc": {"type": "geo_point"},
        "shape": {"type": "geo_shape"},
    }}


def lat(r):
    k = r.random()
    if k < 0.06:
        return r.choice([90.0, -90.0])
    if k < 0.12:
        return round(r.choice([1, -1]) * (90 - r.random() * 1e-3), 7)
    return round(r.uniform(-90, 90), 6)


def lon(r):
    k = r.random()
    if k < 0.06:
        return r.choice([180.0, -180.0])
    if k < 0.12:
        return round(r.choice([1, -1]) * (180 - r.random() * 1e-2), 7)
    return round(r.uniform(-180, 180), 6)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def polygon(r, near_lat=None, near_lon=None, hole=False):
    """A ring of 4-10 vertices around a centre, counter-clockwise, as GeoJSON."""
    c_lat = near_lat if near_lat is not None else r.uniform(-80, 80)
    c_lon = near_lon if near_lon is not None else r.uniform(-170, 170)
    rad = 10 ** r.uniform(-1.5, 1)
    n = r.randint(4, 10)
    import math
    ring = []
    for i in range(n):
        a = 2 * math.pi * i / n
        ring.append([round(clamp(c_lon + rad * math.cos(a), -180, 180), 6), round(clamp(c_lat + rad * math.sin(a), -90, 90), 6)])
    ring.append(ring[0])
    rings = [ring]
    if hole:
        h = rad / 4
        rings.append([[round(c_lon - h, 6), round(c_lat - h, 6)], [round(c_lon - h, 6), round(c_lat + h, 6)],
                      [round(c_lon + h, 6), round(c_lat + h, 6)], [round(c_lon + h, 6), round(c_lat - h, 6)],
                      [round(c_lon - h, 6), round(c_lat - h, 6)]])
    return {"type": "polygon", "coordinates": rings}


def shape(r):
    k = r.randint(0, 9)
    if k <= 1:
        return {"type": "point", "coordinates": [lon(r), lat(r)]}
    if k <= 3:
        la, lo = r.uniform(-85, 85), r.uniform(-175, 175)
        return {"type": "linestring", "coordinates": [[round(clamp(lo + r.gauss(0, 2), -180, 180), 6), round(clamp(la + r.gauss(0, 2), -90, 90), 6)] for _ in range(r.randint(2, 5))]}
    if k == 4:
        # A box across the antimeridian (OpenSearch splits it in two).
        la = r.uniform(-60, 60)
        return {"type": "envelope", "coordinates": [[r.uniform(170, 179), la + 2], [r.uniform(-179, -170), la - 2]]}
    if k == 5:
        return {"type": "multipoint", "coordinates": [[lon(r), lat(r)] for _ in range(r.randint(2, 4))]}
    if k == 6:
        # A polygon at a pole.
        return polygon(r, near_lat=r.choice([88.5, -88.5]), hole=r.random() < 0.3)
    return polygon(r, hole=r.random() < 0.3)


def load(req, check, index, docs, seed):
    r = random.Random(seed)
    batch = max(1, docs // 6)
    for start in range(0, docs, batch):
        lines = []
        for i in range(start, min(docs, start + batch)):
            doc = {"body": " ".join(r.choice(WORDS) for _ in range(r.randint(1, 8))), "tag": r.choice(WORDS), "n": i}
            k = r.random()
            if k < 0.85:
                pts = [[lon(r), lat(r)] for _ in range(1 if r.random() < 0.8 else r.randint(2, 3))]
                doc["loc"] = pts[0] if len(pts) == 1 else pts
            if r.random() < 0.8:
                doc["shape"] = shape(r)
            # A few large shapes, so that CONTAINS (the indexed shape holds the query's) matches.
            if i % 300 == 0:
                doc["shape"] = {"type": "envelope", "coordinates": [[-75, 75], [75, -75]]}
            elif i % 300 == 1:
                doc["shape"] = {"type": "polygon", "coordinates": [[[-180, 80], [180, 80], [180, 90], [-180, 90], [-180, 80]]]}
            lines.append(json.dumps({"index": {"_index": index, "_id": str(i)}}))
            lines.append(json.dumps(doc))
        res = req("POST", "/_bulk", "\n".join(lines) + "\n", ndjson=True)
        bad = [it for it in res["items"] if "error" in it["index"]]
        # OpenSearch may refuse a random shape (a self-intersecting ring); only those.
        check(len(bad) < batch // 20, f"{index}: geo bulk batch at {start}: {len(bad)} refused, e.g. {bad[:1]}")
        req("POST", f"/{index}/_refresh")
    lines = []
    for i in r.sample(range(docs), docs // 40):
        lines.append(json.dumps({"delete": {"_index": index, "_id": str(i)}}))
    req("POST", "/_bulk", "\n".join(lines) + "\n", ndjson=True)
    req("POST", f"/{index}/_refresh")


def create_shapes(req):
    """The index an indexed_shape query reads its shape from."""
    try:
        req("DELETE", "/geo-shapes")
    except RuntimeError:
        pass
    req("PUT", "/geo-shapes", {"settings": {"number_of_shards": 1, "number_of_replicas": 0},
                               "mappings": {"properties": {"area": {"type": "geo_shape"}}}})
    req("PUT", "/geo-shapes/_doc/square?refresh=true", {"area": {"type": "polygon", "coordinates": [[[-20, -20], [20, -20], [20, 20], [-20, 20], [-20, -20]]]}})
    req("PUT", "/geo-shapes/_doc/dateline?refresh=true", {"area": {"type": "envelope", "coordinates": [[160, 30], [-160, -30]]}})


def rows():
    q = []

    def add(name, body, expect="native"):
        q.append((name, body, expect))

    def bbox(top, left, bottom, right, field="loc"):
        return {"geo_bounding_box": {field: {"top_left": {"lat": top, "lon": left}, "bottom_right": {"lat": bottom, "lon": right}}}}

    def dist(d, lat_, lon_, field="loc", **kw):
        return {"geo_distance": {"distance": d, field: {"lat": lat_, "lon": lon_}, **kw}}

    def gshape(shape_, relation="intersects", field="shape"):
        return {"geo_shape": {field: {"shape": shape_, "relation": relation}}}

    # geo_point: geo_bounding_box (LatLonPoint.newBoxQuery in an IndexOrDocValuesQuery).
    add("bbox point", {"query": bbox(40, -10, -30, 60)})
    add("bbox point dateline", {"query": bbox(50, 150, -50, -150)})
    add("bbox point poles", {"query": bbox(90, -180, -90, 180), "size": 20})
    add("bbox point north cap", {"query": bbox(90, -180, 85, 180)})
    add("bbox point thin", {"query": bbox(10.000001, -50, 10, 50)})
    add("bbox point wide coerced", {"query": {"geo_bounding_box": {"validation_method": "coerce", "loc": {"top_left": {"lat": 30, "lon": -200}, "bottom_right": {"lat": -30, "lon": 200}}}}})
    # geo_distance (LatLonPoint.newDistanceQuery).
    add("distance point", {"query": dist("2000km", 10, 20)})
    add("distance point pole", {"query": dist("500km", 89.9, 0)})
    add("distance point antimeridian", {"query": dist("1500km", -5, 179.9)})
    add("distance point earth", {"query": dist("20000km", 0, 0), "size": 15})
    add("distance point 1m", {"query": dist("1m", 90, 0)})
    add("distance point plane", {"query": dist("3000km", 40, -100, distance_type="plane")})
    # geo_polygon (LatLonPoint.newPolygonQuery).
    add("polygon point", {"query": {"geo_polygon": {"loc": {"points": [{"lat": 0, "lon": 0}, {"lat": 60, "lon": 10}, {"lat": 10, "lon": 80}]}}}})
    add("polygon point pole", {"query": {"geo_polygon": {"loc": {"points": [[-180, 80], [-60, 89.99], [60, 89.99], [180, 80], [0, 70]]}}}})
    add("polygon point flat", {"query": {"geo_polygon": {"loc": {"points": [[0, 0], [10, 0], [20, 0], [10, 0.0000001]]}}}})
    # geo_shape on a geo_point (INTERSECTS only).
    add("shape on point envelope", {"query": gshape({"type": "envelope", "coordinates": [[-30, 40], [30, -40]]}, field="loc")})
    add("shape on point envelope dateline", {"query": gshape({"type": "envelope", "coordinates": [[170, 20], [-170, -20]]}, field="loc")})
    add("shape on point polygon hole", {"query": gshape({"type": "polygon", "coordinates": [[[-40, -40], [40, -40], [40, 40], [-40, 40], [-40, -40]], [[-10, -10], [10, -10], [10, 10], [-10, 10], [-10, -10]]]}, field="loc")})
    add("shape on point multipolygon", {"query": gshape({"type": "multipolygon", "coordinates": [[[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]], [[[100, -60], [140, -60], [140, -20], [100, -20], [100, -60]]]]}, field="loc")})
    add("shape on point polygon dateline", {"query": gshape({"type": "polygon", "coordinates": [[[170, -10], [-170, -10], [-170, 10], [170, 10], [170, -10]]]}, field="loc")})
    add("shape on point circle", {"query": gshape({"type": "circle", "coordinates": [30, 30], "radius": "800km"}, field="loc")})
    add("shape on point collection", {"query": gshape({"type": "geometrycollection", "geometries": [
        {"type": "envelope", "coordinates": [[-10, 10], [10, -10]]}, {"type": "polygon", "coordinates": [[[-50, -50], [50, -50], [50, 50], [-50, 50], [-50, -50]]]}]}, field="loc")})
    add("shape on point indexed", {"query": {"geo_shape": {"loc": {"indexed_shape": {"index": "geo-shapes", "id": "square", "path": "area"}}}}})
    # geo_shape fields: every relation, every geometry (LatLonShape.newGeometryQuery).
    for rel in ("intersects", "within", "disjoint", "contains"):
        add(f"shape {rel} envelope", {"query": gshape({"type": "envelope", "coordinates": [[-60, 50], [60, -50]]}, rel)})
        add(f"shape {rel} envelope dateline", {"query": gshape({"type": "envelope", "coordinates": [[160, 40], [-160, -40]]}, rel)})
        add(f"shape {rel} polygon", {"query": gshape({"type": "polygon", "coordinates": [[[-30, -30], [30, -30], [30, 30], [-30, 30], [-30, -30]], [[-5, -5], [5, -5], [5, 5], [-5, 5], [-5, -5]]]}, rel)})
        add(f"shape {rel} polygon pole", {"query": gshape({"type": "polygon", "coordinates": [[[-180, 85], [-90, 85], [0, 85], [90, 85], [180, 85], [180, 90], [-180, 90], [-180, 85]]]}, rel)})
        add(f"shape {rel} point", {"query": gshape({"type": "point", "coordinates": [0.5, 0.5]}, rel)})
        add(f"shape {rel} multipoint", {"query": gshape({"type": "multipoint", "coordinates": [[0.5, 0.5], [100, -30]]}, rel)})
        add(f"shape {rel} circle", {"query": gshape({"type": "circle", "coordinates": [-100, 40], "radius": "1500km"}, rel)})
        if rel != "within":
            add(f"shape {rel} linestring", {"query": gshape({"type": "linestring", "coordinates": [[-170, -60], [0, 0], [170, 60]]}, rel)})
            add(f"shape {rel} multilinestring dateline", {"query": gshape({"type": "multilinestring", "coordinates": [[[170, 0], [-170, 5]], [[0, 80], [10, 89]]]}, rel)})
    add("shape indexed dateline", {"query": {"geo_shape": {"shape": {"indexed_shape": {"index": "geo-shapes", "id": "dateline", "path": "area"}, "relation": "intersects"}}}})
    add("shape collection", {"query": gshape({"type": "geometrycollection", "geometries": [
        {"type": "point", "coordinates": [10, 10]}, {"type": "linestring", "coordinates": [[-10, -10], [20, 30]]}]})})
    add("bbox shape", {"query": bbox(40, -10, -30, 60, field="shape")})
    add("bbox shape dateline", {"query": bbox(50, 150, -50, -150, field="shape")})
    add("distance shape", {"query": dist("1000km", 45, 45, field="shape")})
    add("distance shape pole", {"query": dist("300km", -90, 0, field="shape")})
    # In bool queries, as filters beside scored clauses, and boosted.
    add("bool match + bbox filter", {"query": {"bool": {"must": [{"match": {"body": "alpha"}}], "filter": [bbox(60, -120, -60, 120)]}}})
    add("bool match + distance filter", {"query": {"bool": {"must": [{"match": {"body": "beta gamma"}}], "filter": [dist("5000km", 0, 0)]}}})
    add("bool should distances", {"query": {"bool": {"should": [dist("1000km", 0, 0), dist("1000km", 50, 50), {"match": {"body": "delta"}}]}}})
    add("bool shape must_not", {"query": {"bool": {"must": [{"match": {"body": "alpha"}}], "must_not": [gshape({"type": "envelope", "coordinates": [[-90, 45], [90, -45]]})]}}})
    add("boosted distance", {"query": {"geo_distance": {"distance": "3000km", "loc": [10, 10], "boost": 3.5}}})
    add("constant_score shape", {"query": {"constant_score": {"filter": gshape({"type": "envelope", "coordinates": [[0, 60], [60, 0]]}), "boost": 2}}})
    add("geo size 0", {"size": 0, "query": dist("4000km", 20, 20)})
    add("geo track_total_hits 50", {"track_total_hits": 50, "query": bbox(80, -170, -80, 170)})
    add("geo sorted by n", {"query": bbox(60, -120, -60, 120), "sort": [{"n": "desc"}]})
    add("geo terms agg", {"size": 0, "query": dist("6000km", 0, 0), "aggs": {"t": {"terms": {"field": "tag"}}}})

    # _geo_distance sorts. One origin, metres, ascending, min: Lucene's LatLonPointSortField;
    # anything else: OpenSearch's own comparator (GeoDistanceSortBuilder), ARC natively.
    def gsort(origin, **kw):
        return {"_geo_distance": {"loc": origin, **kw}}

    add("geo sort", {"query": {"match": {"body": "alpha"}}, "sort": [gsort([20, 10])]})
    add("geo sort match_all", {"sort": [gsort({"lat": -45, "lon": 100})], "size": 25})
    add("geo sort pole", {"query": dist("3000km", 90, 0), "sort": [gsort([0, 90])]})
    add("geo sort antimeridian", {"query": bbox(60, 150, -60, -150), "sort": [gsort([180, 0]), "_doc"]})
    add("geo sort then n", {"query": {"match": {"body": "beta"}}, "sort": [gsort([0, 0]), {"n": "asc"}]})
    add("geo sort after n", {"query": {"match": {"body": "gamma"}}, "sort": [{"tag": "asc"}, gsort([-70, 40])]})
    add("geo sort track_scores", {"query": {"match": {"body": "delta"}}, "track_scores": True, "sort": [gsort([10, 10])]})
    add("geo sort search_after", {"query": {"match": {"body": "alpha"}}, "sort": [gsort([20, 10]), {"n": "asc"}], "search_after": [5_000_000.0, 100]},
        "search_after_geo")
    add("geo sort desc", {"query": {"match": {"body": "alpha"}}, "sort": [gsort([20, 10], order="desc")]})
    add("geo sort km", {"query": {"match": {"body": "beta"}}, "sort": [gsort([20, 10], unit="km")]})
    add("geo sort miles desc", {"sort": [gsort([-120, 35], unit="mi", order="desc"), "_doc"]})
    # (OpenSearch refuses `sum` for a geo distance.)
    for mode in ("min", "max", "avg", "median"):
        add(f"geo sort mode {mode}", {"query": {"match": {"body": "eta"}}, "sort": [gsort([5, 5], mode=mode), {"n": "desc"}]})
    add("geo sort origins", {"query": {"match": {"body": "theta"}}, "sort": [gsort([[0, 0], [100, -30], [-150, 60]])]})
    add("geo sort origins median desc", {"query": {"match": {"body": "alpha"}}, "sort": [gsort([[0, 89], [179.9, 0]], mode="median", order="desc")]})
    add("geo sort origins search_after", {"query": {"match": {"body": "beta"}}, "sort": [gsort([[0, 0], [90, 0]], unit="km"), {"n": "asc"}], "search_after": [3000.0, 50]})
    # PLANE goes through Math.cos's HotSpot intrinsic: OpenSearch's comparator keeps it. (A plain
    # one-origin ascending sort is Lucene's whatever its distance_type: still native.)
    add("geo sort plane", {"query": {"match": {"body": "alpha"}}, "sort": [gsort([20, 10], distance_type="plane", order="desc")]}, "sort_geo_plane")
    add("geo sort plane as lucene", {"query": {"match": {"body": "alpha"}}, "sort": [gsort([20, 10], distance_type="plane")]})
    add("geo sort unmapped", {"query": {"match": {"body": "alpha"}}, "sort": [{"_geo_distance": {"nowhere": [0, 0], "ignore_unmapped": True}}, {"n": "asc"}]})
    add("geo sort shape filter", {"query": {"bool": {"filter": [gshape({"type": "envelope", "coordinates": [[-60, 50], [60, -50]]})]}}, "sort": [gsort([0, 0], order="desc", unit="km")]})
    return q


def create_legacy(req):
    """A geo_shape field on the deprecated prefix-tree mapping (`tree`): its
    queries are spatial-extras' (RecursivePrefixTreeStrategy), which the
    plugin does not route, so each falls back by name."""
    try:
        req("DELETE", "/geo-legacy")
    except RuntimeError:
        pass
    req("PUT", "/geo-legacy", {"settings": {"number_of_shards": 1, "number_of_replicas": 0},
                               "mappings": {"properties": {"shape": {"type": "geo_shape", "tree": "quadtree", "precision": "100km"}}}})
    r = random.Random(5)
    lines = []
    for i in range(200):
        lines.append(json.dumps({"index": {"_index": "geo-legacy", "_id": str(i)}}))
        lines.append(json.dumps({"shape": {"type": "point", "coordinates": [round(r.uniform(-180, 180), 4), round(r.uniform(-90, 90), 4)]}}))
    req("POST", "/_bulk?refresh=true", "\n".join(lines) + "\n", ndjson=True)


def legacy_rows():
    return [
        ("legacy shape intersects", {"query": {"geo_shape": {"shape": {"shape": {"type": "envelope", "coordinates": [[-60, 60], [60, -60]]}}}}},
         "clause_IntersectsPrefixTreeQuery"),
    ]
