/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.apache.lucene.document.LatLonPoint;
import org.apache.lucene.document.ShapeField;
import org.apache.lucene.geo.Circle;
import org.apache.lucene.geo.Line;
import org.apache.lucene.geo.Point;
import org.apache.lucene.geo.Polygon;
import org.apache.lucene.geo.Rectangle;
import org.apache.lucene.search.PointRangeQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.SortField;
import org.apache.lucene.util.SloppyMath;
import org.opensearch.common.geo.GeoDistance;
import org.opensearch.common.geo.GeoPoint;
import org.opensearch.index.fielddata.IndexFieldData;
import org.opensearch.index.fielddata.plain.AbstractLatLonPointIndexFieldData.LatLonPointIndexFieldData;
import org.opensearch.search.MultiValueMode;
import org.opensearch.search.sort.GeoDistanceSortBuilder;

import java.io.ByteArrayOutputStream;
import java.lang.reflect.Field;
import java.nio.charset.StandardCharsets;

/**
 * The geo half of {@link QueryEncoder}: the Lucene queries OpenSearch builds for {@code
 * geo_bounding_box}, {@code geo_distance}, {@code geo_polygon} and {@code geo_shape} (M9 T9.6), as
 * the nodes {@code decode_node} in {@code crates/lucene-ffi/src/jvm_reader.rs} reads.
 *
 * <ul>
 *   <li>{@code LatLonPoint.newBoxQuery}: its two-dimension {@link PointRangeQuery} (an anonymous
 *       subclass declared in {@link LatLonPoint}) -- a {@code geo_point} box; across the dateline
 *       Lucene builds a {@code ConstantScoreQuery} of two, which the tree encodes as it is;
 *   <li>{@code LatLonPointDistanceQuery} -- {@code geo_distance}, and a {@code geo_shape} circle, on
 *       a {@code geo_point};
 *   <li>{@code LatLonPointQuery} -- {@code geo_polygon}, and a {@code geo_shape} polygon on a {@code
 *       geo_point};
 *   <li>{@code LatLonShapeQuery} and {@code LatLonShapeBoundingBoxQuery} -- every query on a {@code
 *       geo_shape} field ({@code LatLonShape.newGeometryQuery}), under every relation.
 * </ul>
 *
 * <p>The four query classes are package-private, so they are recognised by name and read through
 * {@link Reflect}: the fields their constructors were given, which are all their {@code equals}
 * compares. The geometries are re-validated by the native constructors. {@code IndexOrDocValuesQuery}
 * around a {@code geo_point} query is unwrapped by {@link QueryEncoder} to its index side, as for
 * ranges: the two sides match the same documents.
 *
 * <p>Not encoded (each falls back under a named reason): a points range of another width
 * ({@code points_width}), a geometry class Lucene's factories do not build ({@code geo_geometry}),
 * and fields the reflection cannot read ({@code geo_reflect}) -- a Lucene whose classes changed.
 */
final class GeoEncoder {
    static final byte NODE_POINTS_BOX = 15;
    static final byte NODE_GEO_DISTANCE = 16;
    static final byte NODE_GEO_POINT = 17;
    static final byte NODE_GEO_SHAPE = 18;
    static final byte NODE_GEO_SHAPE_BOX = 19;

    static final byte GEOMETRY_POINT = 0;
    static final byte GEOMETRY_LINE = 1;
    static final byte GEOMETRY_POLYGON = 2;
    static final byte GEOMETRY_RECTANGLE = 3;
    static final byte GEOMETRY_CIRCLE = 4;

    /** What {@link #node} returns for a query that is not a geo query. */
    static final String NOT_GEO = "not_geo";

    private static final String DOCUMENT = "org.apache.lucene.document.";
    private static final String DISTANCE = DOCUMENT + "LatLonPointDistanceQuery";
    private static final String POINT = DOCUMENT + "LatLonPointQuery";
    private static final String SHAPE = DOCUMENT + "LatLonShapeQuery";
    private static final String SHAPE_BOX = DOCUMENT + "LatLonShapeBoundingBoxQuery";

    private GeoEncoder() {}

    /**
     * Appends {@code q} when it is a geo query: null when it was written, a fallback reason when it
     * is a geo query that cannot be, {@link #NOT_GEO} when it is none.
     */
    static String node(Query q, ByteArrayOutputStream out) {
        if (q instanceof PointRangeQuery pr) {
            return pointsBox(pr, out);
        }
        String name = q.getClass().getName();
        try {
            return switch (name) {
                case DISTANCE -> distance(q, out);
                case POINT -> spatial(q, NODE_GEO_POINT, out);
                case SHAPE -> spatial(q, NODE_GEO_SHAPE, out);
                case SHAPE_BOX -> spatial(q, NODE_GEO_SHAPE_BOX, out);
                default -> NOT_GEO;
            };
        } catch (ReflectiveOperationException | RuntimeException e) {
            return "geo_reflect";
        }
    }

    /** {@code LatLonPoint.newBoxQuery}'s range; any other multi-dimension or 4-byte range is not. */
    private static String pointsBox(PointRangeQuery pr, ByteArrayOutputStream out) {
        if (pr.getClass().getEnclosingClass() != LatLonPoint.class || pr.getNumDims() != 2 || pr.getBytesPerDim() != Integer.BYTES) {
            return "points_width";
        }
        out.write(NODE_POINTS_BOX);
        writeBytes(out, pr.getField().getBytes(StandardCharsets.UTF_8));
        writeInt(out, pr.getNumDims());
        writeBytes(out, pr.getLowerPoint());
        writeBytes(out, pr.getUpperPoint());
        return null;
    }

    private static String distance(Query q, ByteArrayOutputStream out) throws ReflectiveOperationException {
        Class<?> c = q.getClass();
        Field field = Reflect.declared(c, "field");
        Field lat = Reflect.declared(c, "latitude");
        Field lon = Reflect.declared(c, "longitude");
        Field radius = Reflect.declared(c, "radiusMeters");
        if (field == null || lat == null || lon == null || radius == null) {
            return "geo_reflect";
        }
        out.write(NODE_GEO_DISTANCE);
        writeBytes(out, ((String) field.get(q)).getBytes(StandardCharsets.UTF_8));
        writeDouble(out, lat.getDouble(q));
        writeDouble(out, lon.getDouble(q));
        writeDouble(out, radius.getDouble(q));
        return null;
    }

    /** A {@code SpatialQuery}: its field, relation and geometries. */
    private static String spatial(Query q, byte kind, ByteArrayOutputStream out) throws ReflectiveOperationException {
        Field field = Reflect.inHierarchy(q.getClass(), "field");
        Field relation = Reflect.inHierarchy(q.getClass(), "queryRelation");
        Field geometries = Reflect.inHierarchy(q.getClass(), "geometries");
        if (field == null || relation == null || geometries == null) {
            return "geo_reflect";
        }
        Object[] geoms = (Object[]) geometries.get(q);
        ShapeField.QueryRelation rel = (ShapeField.QueryRelation) relation.get(q);
        ByteArrayOutputStream body = new ByteArrayOutputStream();
        body.write(kind);
        writeBytes(body, ((String) field.get(q)).getBytes(StandardCharsets.UTF_8));
        body.write(rel.ordinal());
        if (kind == NODE_GEO_SHAPE_BOX) {
            if (geoms.length != 1 || geoms[0].getClass() != Rectangle.class) {
                return "geo_geometry";
            }
            Rectangle r = (Rectangle) geoms[0];
            writeDouble(body, r.minLat);
            writeDouble(body, r.maxLat);
            writeDouble(body, r.minLon);
            writeDouble(body, r.maxLon);
        } else {
            if (geoms.length == 0) {
                return "geo_geometry";
            }
            writeInt(body, geoms.length);
            for (Object g : geoms) {
                if (geometry(g, body) == false) {
                    return "geo_geometry";
                }
            }
        }
        out.writeBytes(body.toByteArray());
        return null;
    }

    /** One geometry, as {@code decode_geometry} reads it; false for a class it has no tag for. */
    private static boolean geometry(Object g, ByteArrayOutputStream out) {
        Class<?> c = g.getClass();
        if (c == Point.class) {
            Point p = (Point) g;
            out.write(GEOMETRY_POINT);
            writeDouble(out, p.getLat());
            writeDouble(out, p.getLon());
        } else if (c == Line.class) {
            Line l = (Line) g;
            out.write(GEOMETRY_LINE);
            vertices(out, l.getLats(), l.getLons());
        } else if (c == Polygon.class) {
            out.write(GEOMETRY_POLYGON);
            polygon((Polygon) g, out);
        } else if (c == Rectangle.class) {
            Rectangle r = (Rectangle) g;
            out.write(GEOMETRY_RECTANGLE);
            writeDouble(out, r.minLat);
            writeDouble(out, r.maxLat);
            writeDouble(out, r.minLon);
            writeDouble(out, r.maxLon);
        } else if (c == Circle.class) {
            Circle ci = (Circle) g;
            out.write(GEOMETRY_CIRCLE);
            writeDouble(out, ci.getLat());
            writeDouble(out, ci.getLon());
            writeDouble(out, ci.getRadius());
        } else {
            return false;
        }
        return true;
    }

    private static void polygon(Polygon p, ByteArrayOutputStream out) {
        vertices(out, p.getPolyLats(), p.getPolyLons());
        Polygon[] holes = p.getHoles();
        writeInt(out, holes.length);
        for (Polygon h : holes) {
            polygon(h, out);
        }
    }

    private static void vertices(ByteArrayOutputStream out, double[] lats, double[] lons) {
        writeInt(out, lats.length);
        for (double v : lats) {
            writeDouble(out, v);
        }
        for (double v : lons) {
            writeDouble(out, v);
        }
    }

    // --- _geo_distance sorts ----------------------------------------------------------------

    private static final String LUCENE_SORT = DOCUMENT + "LatLonPointSortField";

    /** Whether {@code f} is Lucene's {@code LatLonPointSortField} (OpenSearch's single-origin, metres, ascending {@code _geo_distance}). */
    static boolean luceneSort(SortField f) {
        return f.getClass().getName().equals(LUCENE_SORT);
    }

    /** Whether {@code f} is a key of OpenSearch's own {@code _geo_distance} comparator source. */
    static boolean openSearchSort(SortField f) {
        return f.getClass() == SortField.class
            && f.getType() == SortField.Type.CUSTOM
            && f.getComparatorSource() != null
            && f.getComparatorSource().getClass().getEnclosingClass() == GeoDistanceSortBuilder.class;
    }

    /**
     * A {@code _geo_distance} key as {@code decode_geo_sort} reads it: {@code lucene} for {@code
     * LatLonPointSortField}, else OpenSearch's comparator with its origins, unit (metres per unit)
     * and mode.
     */
    record Sort(boolean lucene, String field, double[] lats, double[] lons, double unitMeters, MultiValueMode mode) {}

    /** At most this many origins ({@code MAX_GEO_ORIGINS} in {@code jvm_reader.rs}). */
    static final int MAX_ORIGINS = 1024;

    /**
     * {@code f}'s geo sort, or the reason it cannot run natively ({@code String}); null when {@code f}
     * is no geo sort. OpenSearch's comparator is taken when it is not nested, measures {@code ARC}
     * (not {@code PLANE}: {@code Math.cos} is a HotSpot intrinsic no portable code matches bit for
     * bit), reads a {@code geo_point}'s own doc values, and has finite origins.
     */
    static Object sort(SortField f) {
        try {
            if (luceneSort(f)) {
                Field lat = Reflect.declared(f.getClass(), "latitude");
                Field lon = Reflect.declared(f.getClass(), "longitude");
                if (lat == null || lon == null) {
                    return "sort_geo_reflect";
                }
                return new Sort(true, f.getField(), new double[] { lat.getDouble(f) }, new double[] { lon.getDouble(f) }, 1.0, MultiValueMode.MIN);
            }
            if (openSearchSort(f) == false) {
                return null;
            }
            IndexFieldData.XFieldComparatorSource src = (IndexFieldData.XFieldComparatorSource) f.getComparatorSource();
            if (src.nested() != null) {
                return "sort_geo_nested";
            }
            Class<?> c = src.getClass();
            Field builder = Reflect.declared(c, "this$0");
            Field points = Reflect.declared(c, "val$localPoints");
            Field data = Reflect.declared(c, "val$geoIndexFieldData");
            if (builder == null || points == null || data == null) {
                return "sort_geo_reflect";
            }
            GeoDistanceSortBuilder b = (GeoDistanceSortBuilder) builder.get(src);
            if (b.geoDistance() != GeoDistance.ARC) {
                return "sort_geo_plane";
            }
            if (!(data.get(src) instanceof LatLonPointIndexFieldData ifd)) {
                return "sort_geo_field_data";
            }
            GeoPoint[] origins = (GeoPoint[]) points.get(src);
            if (origins.length == 0 || origins.length > MAX_ORIGINS) {
                return "sort_geo_origins";
            }
            double[] lats = new double[origins.length], lons = new double[origins.length];
            for (int i = 0; i < origins.length; i++) {
                lats[i] = origins[i].lat();
                lons[i] = origins[i].lon();
                if (Double.isFinite(lats[i]) == false || Double.isFinite(lons[i]) == false) {
                    return "sort_geo_origins";
                }
            }
            return new Sort(false, ifd.getFieldName(), lats, lons, b.unit().toMeters(1.0), src.sortMode());
        } catch (ReflectiveOperationException | RuntimeException e) {
            return "sort_geo_reflect";
        }
    }

    /** {@code SloppyMath.haversinMeters} of a sort key, as {@code LatLonPointDistanceComparator.haversin2}. */
    static double haversin2(double key) {
        return Double.isInfinite(key) ? key : SloppyMath.haversinMeters(key);
    }

    private static void writeDouble(ByteArrayOutputStream out, double v) {
        long bits = Double.doubleToRawLongBits(v);
        writeInt(out, (int) bits);
        writeInt(out, (int) (bits >>> 32));
    }

    private static void writeBytes(ByteArrayOutputStream out, byte[] b) {
        writeInt(out, b.length);
        out.writeBytes(b);
    }

    private static void writeInt(ByteArrayOutputStream out, int v) {
        out.write(v);
        out.write(v >>> 8);
        out.write(v >>> 16);
        out.write(v >>> 24);
    }
}
