import java.util.ArrayList;
import java.util.List;
import org.apache.lucene.facet.taxonomy.FacetLabel;
import org.apache.lucene.facet.taxonomy.directory.DirectoryTaxonomyReader;
import org.apache.lucene.facet.taxonomy.directory.DirectoryTaxonomyWriter;
import org.apache.lucene.facet.taxonomy.writercache.LruTaxonomyWriterCache;
import org.apache.lucene.index.IndexWriterConfig.OpenMode;
import org.apache.lucene.store.ByteBuffersDirectory;

/**
 * M13 T13.4a's taxonomy pair (the Rust twin is {@code
 * benchmarks/rust-runner/src/micro_taxonomy.rs}) over 20,000 labels {@code dim<i%10> / l<(i/10)%50>
 * / v<i>} (20,511 categories with their ancestors). {@code taxo_add_new}: a fresh {@code
 * DirectoryTaxonomyWriter} (default cache of 4000) adding every label, units: labels; {@code
 * taxo_add_cached}: the same labels again on a writer whose unbounded cache holds them all, units:
 * labels; {@code taxo_add_lookup}: the same labels again on a writer reopened over the committed
 * taxonomy with a cache of 1000 that is never filled, so most adds look the category up in the
 * writer's near-real-time reader, units: labels; {@code taxo_ordinal}: {@code
 * DirectoryTaxonomyReader.getOrdinal} of every label in order (its 4000-entry cache misses each),
 * units: lookups; {@code taxo_path}: {@code getPath} of every ordinal, units: lookups. Output is
 * {@code name	ns_per_unit	units}.
 */
public class TaxonomyMicro {
  static List<FacetLabel> labels() {
    List<FacetLabel> out = new ArrayList<>();
    for (int i = 0; i < 20000; i++) {
      out.add(new FacetLabel("dim" + (i % 10), "l" + ((i / 10) % 50), "v" + i));
    }
    return out;
  }

  public static void main(String[] args) throws Exception {
    List<FacetLabel> labels = labels();

    SweepMicro.measure("taxo_add_new", () -> {
      try (DirectoryTaxonomyWriter w = new DirectoryTaxonomyWriter(new ByteBuffersDirectory())) {
        for (FacetLabel l : labels) SweepMicro.sink += w.addCategory(l);
      }
      return labels.size();
    });

    ByteBuffersDirectory dir = new ByteBuffersDirectory();
    try (DirectoryTaxonomyWriter w =
        new DirectoryTaxonomyWriter(dir, OpenMode.CREATE, new LruTaxonomyWriterCache(Integer.MAX_VALUE))) {
      for (FacetLabel l : labels) w.addCategory(l);
      SweepMicro.measure("taxo_add_cached", () -> {
        for (FacetLabel l : labels) SweepMicro.sink += w.addCategory(l);
        return labels.size();
      });
    }

    try (DirectoryTaxonomyWriter w =
        new DirectoryTaxonomyWriter(dir, OpenMode.APPEND, new LruTaxonomyWriterCache(1000))) {
      w.setCacheMissesUntilFill(Integer.MAX_VALUE);
      SweepMicro.measure("taxo_add_lookup", () -> {
        for (FacetLabel l : labels) SweepMicro.sink += w.addCategory(l);
        return labels.size();
      });
    }

    try (DirectoryTaxonomyReader r = new DirectoryTaxonomyReader(dir)) {
      SweepMicro.measure("taxo_ordinal", () -> {
        for (FacetLabel l : labels) SweepMicro.sink += r.getOrdinal(l);
        return labels.size();
      });
      int size = r.getSize();
      SweepMicro.measure("taxo_path", () -> {
        for (int o = 0; o < size; o++) SweepMicro.sink += r.getPath(o).length;
        return size;
      });
    }
  }
}
