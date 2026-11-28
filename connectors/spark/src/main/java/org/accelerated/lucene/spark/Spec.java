package org.accelerated.lucene.spark;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

/** The table an index is read as: {@code name:kind} pairs, kinds key, long, keyword, text, vector:dim. */
final class Spec implements Serializable {
  private static final long serialVersionUID = 1L;

  final String name;
  final List<String> columns = new ArrayList<>();
  final List<String> kinds = new ArrayList<>();

  Spec(String name, String columnSpec) {
    this.name = name;
    for (String part : columnSpec.split(",")) {
      String p = part.trim();
      if (p.isEmpty()) {
        continue;
      }
      int colon = p.indexOf(':');
      if (colon <= 0) {
        throw new IllegalArgumentException("column `" + p + "` needs a kind: name:key|long|keyword|text|vector:dim");
      }
      columns.add(p.substring(0, colon));
      kinds.add(p.substring(colon + 1));
    }
  }

  String json() {
    StringBuilder b = new StringBuilder("{\"name\":").append(Json.quote(name)).append(",\"columns\":[");
    for (int i = 0; i < columns.size(); i++) {
      if (i > 0) {
        b.append(',');
      }
      String kind = kinds.get(i);
      b.append("{\"name\":").append(Json.quote(columns.get(i)));
      if (kind.startsWith("vector:")) {
        b.append(",\"kind\":\"vector\",\"dim\":").append(Integer.parseInt(kind.substring(7)));
      } else {
        b.append(",\"kind\":").append(Json.quote(kind));
      }
      b.append('}');
    }
    return b.append("]}").toString();
  }

  /** The Spark schema of the readable columns (search columns have no values). */
  StructType schema() {
    List<StructField> fields = new ArrayList<>();
    for (int i = 0; i < columns.size(); i++) {
      DataType type = typeOf(kinds.get(i));
      if (type != null) {
        fields.add(DataTypes.createStructField(columns.get(i), type, true));
      }
    }
    return DataTypes.createStructType(fields);
  }

  String kindOf(String column) {
    int i = columns.indexOf(column);
    return i < 0 ? null : kinds.get(i);
  }

  private static DataType typeOf(String kind) {
    switch (kind) {
      case "long":
        return DataTypes.LongType;
      case "key":
      case "keyword":
        return DataTypes.StringType;
      default:
        return null;
    }
  }
}
