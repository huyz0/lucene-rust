import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.facet.FacetField;
import org.apache.lucene.facet.FacetResult;
import org.apache.lucene.facet.Facets;
import org.apache.lucene.facet.FacetsCollector;
import org.apache.lucene.facet.FacetsCollectorManager;
import org.apache.lucene.facet.FacetsConfig;
import org.apache.lucene.facet.LabelAndValue;
import org.apache.lucene.facet.taxonomy.AssociationAggregationFunction;
import org.apache.lucene.facet.taxonomy.FastTaxonomyFacetCounts;
import org.apache.lucene.facet.taxonomy.FloatAssociationFacetField;
import org.apache.lucene.facet.taxonomy.IntAssociationFacetField;
import org.apache.lucene.facet.taxonomy.TaxonomyFacetFloatAssociations;
import org.apache.lucene.facet.taxonomy.TaxonomyFacetIntAssociations;
import org.apache.lucene.facet.taxonomy.directory.DirectoryTaxonomyReader;
import org.apache.lucene.facet.taxonomy.directory.DirectoryTaxonomyWriter;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

/**
 * M13 T13.4b's facet-counting pair (the Rust twin is {@code
 * benchmarks/rust-runner/src/micro_facet_counts.rs}), with the same case names. {@code build <dir>}
 * writes, once, a 200,000-document, four-segment index ({@code dir/index}) and its taxonomy
 * ({@code dir/taxo}, ~4,800 ordinals): a flat {@code Author} (2,000 values), a three-level
 * hierarchical {@code Category}, a multi-valued {@code Tag} (1-3 of 500), int and float
 * associations in their own index fields, and {@code q:rare} on every 100th document. Both engines
 * then read those bytes. Cases (the matching documents collected beforehand, except in {@code
 * facet_collect_all}): {@code facet_collect_all}, a {@code FacetsCollectorManager} search over
 * {@code MatchAllDocsQuery}, units: hits; {@code facet_count_all} and {@code facet_count_rare},
 * {@code FastTaxonomyFacetCounts} over every document (dense counts) and over {@code q:rare} (1%:
 * the sparse hash-table counts), units: hits; {@code facet_top_children}, four {@code
 * getTopChildren(10, ..)} calls over the match-all counts, units: calls; {@code
 * facet_int_assoc_sum} and {@code facet_float_assoc_sum}, the association facets' {@code SUM}
 * over every document, units: hits. Each prints a {@code #check} digest of its result first.
 */
public final class FacetCountsMicro {
  static final int DOCS = 200_000;

  static final class Fnv {
    long h = 0xcbf29ce484222325L;

    void add(long x) {
      h ^= x;
      h *= 0x100000001b3L;
    }

    void str(String s) {
      for (byte b : s.getBytes(StandardCharsets.UTF_8)) add(b);
      add(-1);
    }

    void num(Number n) {
      if (n instanceof Float f) add(Float.floatToIntBits(f));
      else add(n.intValue());
    }

    void result(FacetResult r) {
      if (r == null) {
        add(-2);
        return;
      }
      str(r.dim);
      for (String p : r.path) str(p);
      num(r.value);
      add(r.childCount);
      for (LabelAndValue lv : r.labelValues) {
        str(lv.label);
        num(lv.value);
      }
    }
  }

  static FacetsConfig config() {
    FacetsConfig c = new FacetsConfig();
    c.setHierarchical("Category", true);
    c.setMultiValued("Tag", true);
    c.setIndexFieldName("Score", "$int_assoc");
    c.setMultiValued("Score", true);
    c.setIndexFieldName("Weight", "$float_assoc");
    c.setMultiValued("Weight", true);
    return c;
  }

  static void build(Path dir) throws IOException {
    Random r = new Random(0x13_4b_2026_1018L);
    FacetsConfig config = config();
    IndexWriterConfig cfg = new IndexWriterConfig();
    cfg.setRAMBufferSizeMB(512);
    cfg.setMaxBufferedDocs(IndexWriterConfig.DISABLE_AUTO_FLUSH);
    cfg.setMergePolicy(NoMergePolicy.INSTANCE);
    cfg.setOpenMode(IndexWriterConfig.OpenMode.CREATE);
    try (Directory d = FSDirectory.open(dir.resolve("index"));
        Directory td = FSDirectory.open(dir.resolve("taxo"));
        IndexWriter w = new IndexWriter(d, cfg);
        DirectoryTaxonomyWriter tw = new DirectoryTaxonomyWriter(td)) {
      for (int i = 0; i < DOCS; i++) {
        Document doc = new Document();
        doc.add(new StringField("q", i % 100 == 0 ? "rare" : "common", Field.Store.NO));
        doc.add(new FacetField("Author", "a" + r.nextInt(2000)));
        doc.add(
            new FacetField(
                "Category", "c" + r.nextInt(10), "s" + r.nextInt(20), "l" + r.nextInt(10)));
        int tags = 1 + r.nextInt(3);
        for (int t = 0; t < tags; t++) doc.add(new FacetField("Tag", "t" + r.nextInt(500)));
        doc.add(new IntAssociationFacetField(r.nextInt(100), "Score", "k" + r.nextInt(50)));
        doc.add(new FloatAssociationFacetField(r.nextFloat(), "Weight", "k" + r.nextInt(50)));
        w.addDocument(config.build(tw, doc));
        if ((i + 1) % (DOCS / 4) == 0) w.commit();
      }
      tw.commit();
    }
    // Last: its presence is what marks the corpus built.
    Files.writeString(dir.resolve("built"), "ok\n");
  }

  interface Case {
    long run(Fnv f) throws IOException;
  }

  static void cases(String name, Case c) throws IOException {
    Fnv f = new Fnv();
    long units = c.run(f);
    System.out.printf("#check\t%s\t%016x\t%d%n", name, f.h, units);
    SweepMicro.measure(
        name,
        () -> {
          Fnv g = new Fnv();
          long n = c.run(g);
          SweepMicro.sink += g.h;
          return n;
        });
  }

  static void tops(Fnv f, Facets facets) throws IOException {
    f.result(facets.getTopChildren(10, "Author"));
    f.result(facets.getTopChildren(10, "Category"));
    f.result(facets.getTopChildren(10, "Category", "c3"));
    f.result(facets.getTopChildren(10, "Tag"));
  }

  static void run(Path dir) throws IOException {
    FacetsConfig config = config();
    try (Directory d = FSDirectory.open(dir.resolve("index"));
        Directory td = FSDirectory.open(dir.resolve("taxo"));
        DirectoryReader reader = DirectoryReader.open(d);
        DirectoryTaxonomyReader taxo = new DirectoryTaxonomyReader(td)) {
      IndexSearcher searcher = new IndexSearcher(reader);
      Query all = new MatchAllDocsQuery();
      Query rare = new TermQuery(new Term("q", "rare"));
      FacetsCollector fcAll = searcher.search(all, new FacetsCollectorManager());
      FacetsCollector fcRare = searcher.search(rare, new FacetsCollectorManager());
      long hitsAll = hits(fcAll);
      long hitsRare = hits(fcRare);
      AssociationAggregationFunction sum = AssociationAggregationFunction.SUM;

      cases(
          "facet_collect_all",
          f -> {
            FacetsCollector fc = searcher.search(all, new FacetsCollectorManager());
            long n = hits(fc);
            f.add(n);
            return n;
          });
      cases(
          "facet_count_all",
          f -> {
            Facets c = new FastTaxonomyFacetCounts(taxo, config, fcAll);
            f.result(c.getTopChildren(10, "Author"));
            return hitsAll;
          });
      cases(
          "facet_count_rare",
          f -> {
            Facets c = new FastTaxonomyFacetCounts(taxo, config, fcRare);
            f.result(c.getTopChildren(10, "Author"));
            return hitsRare;
          });
      Facets counts = new FastTaxonomyFacetCounts(taxo, config, fcAll);
      cases(
          "facet_top_children",
          f -> {
            tops(f, counts);
            return 4;
          });
      cases(
          "facet_int_assoc_sum",
          f -> {
            Facets c = new TaxonomyFacetIntAssociations("$int_assoc", taxo, config, fcAll, sum);
            f.result(c.getTopChildren(10, "Score"));
            return hitsAll;
          });
      cases(
          "facet_float_assoc_sum",
          f -> {
            Facets c = new TaxonomyFacetFloatAssociations("$float_assoc", taxo, config, fcAll, sum);
            f.result(c.getTopChildren(10, "Weight"));
            return hitsAll;
          });
    }
  }

  static long hits(FacetsCollector fc) {
    long n = 0;
    for (FacetsCollector.MatchingDocs m : fc.getMatchingDocs()) n += m.totalHits();
    return n;
  }

  public static void main(String[] args) throws IOException {
    Path dir = Path.of(args[1]);
    if (args[0].equals("build")) {
      if (Files.exists(dir.resolve("built"))) return;
      Files.createDirectories(dir);
      build(dir);
    } else {
      run(dir);
    }
  }
}
