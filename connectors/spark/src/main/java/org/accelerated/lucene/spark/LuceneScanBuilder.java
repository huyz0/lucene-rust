package org.accelerated.lucene.spark;

import java.util.ArrayList;
import java.util.List;
import org.apache.spark.sql.connector.read.Scan;
import org.apache.spark.sql.connector.read.ScanBuilder;
import org.apache.spark.sql.connector.read.SupportsPushDownFilters;
import org.apache.spark.sql.connector.read.SupportsPushDownLimit;
import org.apache.spark.sql.connector.read.SupportsPushDownRequiredColumns;
import org.apache.spark.sql.sources.And;
import org.apache.spark.sql.sources.EqualTo;
import org.apache.spark.sql.sources.Filter;
import org.apache.spark.sql.sources.GreaterThan;
import org.apache.spark.sql.sources.GreaterThanOrEqual;
import org.apache.spark.sql.sources.In;
import org.apache.spark.sql.sources.IsNotNull;
import org.apache.spark.sql.sources.IsNull;
import org.apache.spark.sql.sources.LessThan;
import org.apache.spark.sql.sources.LessThanOrEqual;
import org.apache.spark.sql.sources.Not;
import org.apache.spark.sql.types.StructType;

/** Takes the columns, the comparisons it can answer and the limit into the index scan. */
final class LuceneScanBuilder
    implements ScanBuilder, SupportsPushDownRequiredColumns, SupportsPushDownFilters, SupportsPushDownLimit {
  private final Options options;
  private StructType required;
  private final List<String> pushed = new ArrayList<>();
  private Filter[] pushedFilters = new Filter[0];
  private int limit = -1;

  LuceneScanBuilder(StructType schema, Options options) {
    this.options = options;
    this.required = schema;
  }

  @Override
  public void pruneColumns(StructType requiredSchema) {
    this.required = requiredSchema;
  }

  @Override
  public Filter[] pushFilters(Filter[] filters) {
    List<Filter> rest = new ArrayList<>();
    List<Filter> taken = new ArrayList<>();
    for (Filter f : filters) {
      List<String> json = new ArrayList<>();
      if (translate(f, json)) {
        pushed.addAll(json);
        taken.add(f);
      } else {
        rest.add(f);
      }
    }
    pushedFilters = taken.toArray(new Filter[0]);
    return rest.toArray(new Filter[0]);
  }

  @Override
  public Filter[] pushedFilters() {
    return pushedFilters;
  }

  @Override
  public boolean pushLimit(int limit) {
    this.limit = limit;
    return true;
  }

  /** Each partition stops at the limit; Spark still limits the union. */
  @Override
  public boolean isPartiallyPushed() {
    return true;
  }

  @Override
  public Scan build() {
    StringBuilder cols = new StringBuilder("[");
    String[] names = required.fieldNames();
    for (int i = 0; i < names.length; i++) {
      if (i > 0) {
        cols.append(',');
      }
      cols.append(Json.quote(names[i]));
    }
    cols.append(']');
    StringBuilder req = new StringBuilder("{\"columns\":").append(cols).append(",\"dictionary\":false,\"filters\":[");
    req.append(String.join(",", pushed)).append(']');
    if (limit >= 0) {
      req.append(",\"limit\":").append(limit);
    }
    if (options.batchRows > 0) {
      req.append(",\"batch_rows\":").append(options.batchRows);
    }
    req.append('}');
    return new LuceneScan(required, options, req.toString());
  }

  private boolean translate(Filter f, List<String> out) {
    if (f instanceof And) {
      List<String> l = new ArrayList<>();
      List<String> r = new ArrayList<>();
      if (translate(((And) f).left(), l) && translate(((And) f).right(), r)) {
        out.addAll(l);
        out.addAll(r);
        return true;
      }
      return false;
    }
    if (f instanceof EqualTo) {
      EqualTo e = (EqualTo) f;
      return cmp(e.attribute(), "=", e.value(), out);
    }
    if (f instanceof Not && ((Not) f).child() instanceof EqualTo) {
      EqualTo e = (EqualTo) ((Not) f).child();
      return cmp(e.attribute(), "!=", e.value(), out);
    }
    if (f instanceof GreaterThan) {
      GreaterThan e = (GreaterThan) f;
      return cmp(e.attribute(), ">", e.value(), out);
    }
    if (f instanceof GreaterThanOrEqual) {
      GreaterThanOrEqual e = (GreaterThanOrEqual) f;
      return cmp(e.attribute(), ">=", e.value(), out);
    }
    if (f instanceof LessThan) {
      LessThan e = (LessThan) f;
      return cmp(e.attribute(), "<", e.value(), out);
    }
    if (f instanceof LessThanOrEqual) {
      LessThanOrEqual e = (LessThanOrEqual) f;
      return cmp(e.attribute(), "<=", e.value(), out);
    }
    if (f instanceof In) {
      In e = (In) f;
      String kind = options.spec.kindOf(e.attribute());
      if (kind == null) {
        return false;
      }
      StringBuilder b = new StringBuilder("{\"column\":").append(Json.quote(e.attribute())).append(",\"in\":[");
      Object[] values = e.values();
      for (int i = 0; i < values.length; i++) {
        String lit = literal(kind, "=", values[i]);
        if (lit == null) {
          return false;
        }
        b.append(i > 0 ? "," : "").append(lit);
      }
      out.add(b.append("]}").toString());
      return true;
    }
    if (f instanceof IsNull || f instanceof IsNotNull) {
      String col = f instanceof IsNull ? ((IsNull) f).attribute() : ((IsNotNull) f).attribute();
      String kind = options.spec.kindOf(col);
      if (kind == null || !(kind.equals("long") || kind.equals("keyword"))) {
        return false;
      }
      out.add("{\"column\":" + Json.quote(col) + ",\"is_null\":" + (f instanceof IsNull) + "}");
      return true;
    }
    return false;
  }

  private boolean cmp(String column, String op, Object value, List<String> out) {
    String kind = options.spec.kindOf(column);
    if (kind == null) {
      return false;
    }
    String lit = literal(kind, op, value);
    if (lit == null) {
      return false;
    }
    out.add("{\"column\":" + Json.quote(column) + ",\"op\":\"" + op + "\",\"value\":" + lit + "}");
    return true;
  }

  /** The JSON of a literal the column's kind can compare with, or null. */
  private static String literal(String kind, String op, Object v) {
    if (kind.equals("long") && (v instanceof Long || v instanceof Integer || v instanceof Short || v instanceof Byte)) {
      return Long.toString(((Number) v).longValue());
    }
    if (kind.equals("keyword") && v instanceof String) {
      return Json.quote((String) v);
    }
    if (kind.equals("key") && v instanceof String && op.equals("=")) {
      return Json.quote((String) v);
    }
    return null;
  }
}
