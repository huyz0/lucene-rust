package org.apache.lucene.document;

import java.io.IOException;
import org.apache.lucene.geo.Component2D;
import org.apache.lucene.geo.LatLonGeometry;
import org.apache.lucene.geo.XYGeometry;
import org.apache.lucene.index.PointValues.Relation;
import org.apache.lucene.search.Query;
import org.apache.lucene.util.BytesRef;

/**
 * Reaches the package-private shape doc-values API for {@code GenGeoShapes} and {@code
 * VerifyGeoShapes}: the {@link LatLonShapeDocValuesQuery} / {@link XYShapeDocValuesQuery}
 * constructors (Lucene's factories only build a box with them) and {@link ShapeDocValues#relate}.
 *
 * <p>It lives in Lucene's own package -- a split package, which the plain classpath allows -- and
 * has no {@code Gen} prefix, so {@code gen-fixtures.sh} compiles it and never runs it.
 */
public final class ShapeAccess {
  private ShapeAccess() {}

  public static Query latLonDocValuesQuery(
      String field, ShapeField.QueryRelation rel, LatLonGeometry... geometries) {
    return new LatLonShapeDocValuesQuery(field, rel, geometries);
  }

  public static Query xyDocValuesQuery(
      String field, ShapeField.QueryRelation rel, XYGeometry... geometries) {
    return new XYShapeDocValuesQuery(field, rel, geometries);
  }

  public static Relation relateLatLon(BytesRef value, Component2D component) throws IOException {
    return new LatLonShapeDocValues(value).relate(component);
  }

  public static Relation relateXY(BytesRef value, Component2D component) throws IOException {
    return new XYShapeDocValues(value).relate(component);
  }

  /** {@code numberOfTerms, encoded minX maxX minY maxY, centroid x y, highest dimension}. */
  public static String header(ShapeDocValues dv) {
    return dv.numberOfTerms()
        + "\t"
        + dv.getEncodedMinX()
        + ","
        + dv.getEncodedMaxX()
        + ","
        + dv.getEncodedMinY()
        + ","
        + dv.getEncodedMaxY()
        + "\t"
        + dv.getEncodedCentroidX()
        + ","
        + dv.getEncodedCentroidY()
        + "\t"
        + dv.getHighestDimension();
  }

  public static ShapeDocValues docValues(ShapeDocValuesField f) {
    return f.shapeDocValues;
  }
}
