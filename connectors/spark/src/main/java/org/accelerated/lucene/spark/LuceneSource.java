package org.accelerated.lucene.spark;

import java.util.Map;
import org.apache.spark.sql.connector.catalog.Table;
import org.apache.spark.sql.connector.catalog.TableProvider;
import org.apache.spark.sql.connector.expressions.Transform;
import org.apache.spark.sql.sources.DataSourceRegister;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.util.CaseInsensitiveStringMap;

/**
 * {@code spark.read.format("lucene").option("path", dir).option("columns", "id:key,ts:long,host:keyword").load()}.
 *
 * <p>Options: {@code path} (the index directory), {@code columns} (the table declaration,
 * {@code name:kind} separated by commas: key, long, keyword, text, vector:dim), {@code table}
 * (its name, default {@code t}), {@code partitions} (default: one per segment up to the
 * default parallelism), {@code batch_rows}.
 */
public final class LuceneSource implements TableProvider, DataSourceRegister {
  @Override
  public String shortName() {
    return "lucene";
  }

  @Override
  public StructType inferSchema(CaseInsensitiveStringMap options) {
    return Options.of(options).spec.schema();
  }

  @Override
  public Table getTable(StructType schema, Transform[] partitioning, Map<String, String> properties) {
    return new LuceneTable(schema, new CaseInsensitiveStringMap(properties));
  }

  @Override
  public boolean supportsExternalMetadata() {
    return true;
  }
}
