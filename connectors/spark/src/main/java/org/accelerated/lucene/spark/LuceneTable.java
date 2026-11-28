package org.accelerated.lucene.spark;

import java.util.Set;
import org.apache.spark.sql.connector.catalog.SupportsRead;
import org.apache.spark.sql.connector.catalog.TableCapability;
import org.apache.spark.sql.connector.read.ScanBuilder;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.util.CaseInsensitiveStringMap;

final class LuceneTable implements SupportsRead {
  private final StructType schema;
  private final Options options;

  LuceneTable(StructType schema, CaseInsensitiveStringMap options) {
    this.schema = schema;
    this.options = Options.of(options);
  }

  @Override
  public String name() {
    return "lucene." + options.spec.name;
  }

  @Override
  public StructType schema() {
    return schema;
  }

  @Override
  public Set<TableCapability> capabilities() {
    return Set.of(TableCapability.BATCH_READ);
  }

  @Override
  public ScanBuilder newScanBuilder(CaseInsensitiveStringMap ignored) {
    return new LuceneScanBuilder(schema, options);
  }
}
