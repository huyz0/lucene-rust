import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.facet.FacetResult;
import org.apache.lucene.facet.Facets;
import org.apache.lucene.facet.FacetsCollector;
import org.apache.lucene.facet.FacetsCollectorManager;
import org.apache.lucene.facet.FacetsConfig;
import org.apache.lucene.facet.RandomSamplingFacetsCollector;
import org.apache.lucene.facet.sortedset.ConcurrentSortedSetDocValuesFacetCounts;
import org.apache.lucene.facet.sortedset.DefaultSortedSetDocValuesReaderState;
import org.apache.lucene.facet.sortedset.SortedSetDocValuesFacetCounts;
import org.apache.lucene.facet.sortedset.SortedSetDocValuesFacetField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

/**
 * M13 T13.4d2's sorted-set facet and sampling pair (the Rust twin is {@code
 * benchmarks/rust-runner/src/micro_ssdv_facets.rs}), with the same case names. {@code build <dir>}
 * writes, once, a 200,000-document, four-segment index of {@code SortedSetDocValuesFacetField}s: a
 * flat {@code Author} (2,000 values), a multi-valued {@code Tag} (1-3 of 500), a hierarchical
 * {@code Date} (year/month), and {@code q:rare} on every 100th document. Both engines then read
 * those bytes. Cases (the hits collected beforehand unless named otherwise): {@code ssdv_state},
 * {@code DefaultSortedSetDocValuesReaderState} built (the ordinal map across the segments), units:
 * labels; {@code ssdv_count_all} and {@code ssdv_count_rare}, {@code
 * SortedSetDocValuesFacetCounts} over every hit (segment counts migrated) and over {@code q:rare}
 * (1%: each ordinal remapped as it comes), then {@code getTopChildren(10, "Author")}, units: hits;
 * {@code ssdv_count_reader}, the counts of every live document, no collector, units: documents;
 * {@code ssdv_count_concurrent}, {@code ConcurrentSortedSetDocValuesFacetCounts} over every hit on
 * a pool of one thread per core (rayon's default size on the Rust side), units: hits; {@code
 * ssdv_top_children}, four {@code getTopChildren(10, ..)} calls over the match-all counts, units:
 * calls; {@code sampling_collect}, a {@code RandomSamplingFacetsCollector} search (sample size
 * 10,000, seed 42) over every document and its sample drawn, units: hits; {@code
 * sampling_count_amortize}, the counts over that sample and {@code amortizeFacetCounts} of {@code
 * Author}, units: sampled hits. Each prints a {@code #check} digest of its result first.
 */
public final class SsdvFacetsMicro {
  static final int DOCS = 200_000;

  static FacetsConfig config() {
    FacetsConfig c = new FacetsConfig();
    c.setHierarchical("Date", true);
    c.setMultiValued("Tag", true);
    return c;
  }

  static void build(Path dir) throws IOException {
    Random r = new Random(0x13_4d_2026_1019L);
    FacetsConfig config = config();
    IndexWriterConfig cfg = new IndexWriterConfig();
    cfg.setRAMBufferSizeMB(512);
    cfg.setMaxBufferedDocs(IndexWriterConfig.DISABLE_AUTO_FLUSH);
    cfg.setMergePolicy(NoMergePolicy.INSTANCE);
    cfg.setOpenMode(IndexWriterConfig.OpenMode.CREATE);
    try (Directory d = FSDirectory.open(dir.resolve("index"));
        IndexWriter w = new IndexWriter(d, cfg)) {
      for (int i = 0; i < DOCS; i++) {
        Document doc = new Document();
        doc.add(new StringField("q", i % 100 == 0 ? "rare" : "common", Field.Store.NO));
        doc.add(new SortedSetDocValuesFacetField("Author", "a" + r.nextInt(2000)));
        int tags = 1 + r.nextInt(3);
        for (int t = 0; t < tags; t++) {
          doc.add(new SortedSetDocValuesFacetField("Tag", "t" + r.nextInt(500)));
        }
        doc.add(
            new SortedSetDocValuesFacetField(
                "Date", "y" + (2000 + r.nextInt(10)), "m" + (1 + r.nextInt(12))));
        w.addDocument(config.build(doc));
        if ((i + 1) % (DOCS / 4) == 0) w.commit();
      }
    }
    // Last: its presence is what marks the corpus built.
    Files.writeString(dir.resolve("built"), "ok\n");
  }

  static void tops(FacetCountsMicro.Fnv f, Facets facets) throws IOException {
    f.result(facets.getTopChildren(10, "Author"));
    f.result(facets.getTopChildren(10, "Tag"));
    f.result(facets.getTopChildren(10, "Date"));
    f.result(facets.getTopChildren(10, "Date", "y2003"));
  }

  static void run(Path dir) throws IOException {
    FacetsConfig config = config();
    ExecutorService exec = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
    try (Directory d = FSDirectory.open(dir.resolve("index"));
        DirectoryReader reader = DirectoryReader.open(d)) {
      IndexSearcher searcher = new IndexSearcher(reader);
      Query all = new MatchAllDocsQuery();
      Query rare = new TermQuery(new Term("q", "rare"));
      FacetsCollector fcAll = searcher.search(all, new FacetsCollectorManager());
      FacetsCollector fcRare = searcher.search(rare, new FacetsCollectorManager());
      long hitsAll = FacetCountsMicro.hits(fcAll);
      long hitsRare = FacetCountsMicro.hits(fcRare);
      DefaultSortedSetDocValuesReaderState state =
          new DefaultSortedSetDocValuesReaderState(reader, config);

      FacetCountsMicro.cases(
          "ssdv_state",
          f -> {
            DefaultSortedSetDocValuesReaderState s =
                new DefaultSortedSetDocValuesReaderState(reader, config);
            f.add(s.getSize());
            return s.getSize();
          });
      FacetCountsMicro.cases(
          "ssdv_count_all",
          f -> {
            Facets c = new SortedSetDocValuesFacetCounts(state, fcAll);
            f.result(c.getTopChildren(10, "Author"));
            return hitsAll;
          });
      FacetCountsMicro.cases(
          "ssdv_count_rare",
          f -> {
            Facets c = new SortedSetDocValuesFacetCounts(state, fcRare);
            f.result(c.getTopChildren(10, "Author"));
            return hitsRare;
          });
      FacetCountsMicro.cases(
          "ssdv_count_reader",
          f -> {
            Facets c = new SortedSetDocValuesFacetCounts(state);
            f.result(c.getTopChildren(10, "Author"));
            return reader.numDocs();
          });
      FacetCountsMicro.cases(
          "ssdv_count_concurrent",
          f -> {
            Facets c;
            try {
              c = new ConcurrentSortedSetDocValuesFacetCounts(state, fcAll, exec);
            } catch (InterruptedException e) {
              throw new IOException(e);
            }
            f.result(c.getTopChildren(10, "Author"));
            return hitsAll;
          });
      Facets counts = new SortedSetDocValuesFacetCounts(state, fcAll);
      FacetCountsMicro.cases(
          "ssdv_top_children",
          f -> {
            tops(f, counts);
            return 4;
          });
      FacetCountsMicro.cases(
          "sampling_collect",
          f -> {
            RandomSamplingFacetsCollector c =
                searcher.search(all, RandomSamplingFacetsCollector.createManager(10_000, 42));
            f.add(sampled(c));
            f.add(Double.doubleToLongBits(c.getSamplingRate()));
            return hitsAll;
          });
      RandomSamplingFacetsCollector sampler =
          searcher.search(all, RandomSamplingFacetsCollector.createManager(10_000, 42));
      long units = sampled(sampler);
      FacetCountsMicro.cases(
          "sampling_count_amortize",
          f -> {
            Facets c = new SortedSetDocValuesFacetCounts(state, sampler);
            FacetResult r = c.getTopChildren(10, "Author");
            f.result(sampler.amortizeFacetCounts(r, config, searcher));
            return units;
          });
    } finally {
      exec.shutdown();
    }
  }

  /** The documents of the sample, iterated. */
  static long sampled(FacetsCollector c) throws IOException {
    long n = 0;
    for (FacetsCollector.MatchingDocs m : c.getMatchingDocs()) {
      DocIdSetIterator it = m.bits().iterator();
      while (it.nextDoc() != DocIdSetIterator.NO_MORE_DOCS) n++;
    }
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
