import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.facet.DrillDownQuery;
import org.apache.lucene.facet.DrillSideways;
import org.apache.lucene.facet.FacetField;
import org.apache.lucene.facet.FacetQuery;
import org.apache.lucene.facet.FacetResult;
import org.apache.lucene.facet.Facets;
import org.apache.lucene.facet.FacetsCollector;
import org.apache.lucene.facet.FacetsCollectorManager;
import org.apache.lucene.facet.FacetsConfig;
import org.apache.lucene.facet.LabelAndValue;
import org.apache.lucene.facet.MultiFacetQuery;
import org.apache.lucene.facet.taxonomy.TaxonomyReader;
import org.apache.lucene.facet.taxonomy.directory.DirectoryTaxonomyReader;
import org.apache.lucene.facet.taxonomy.directory.DirectoryTaxonomyWriter;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.BooleanClause.Occur;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.FieldDoc;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.PhraseQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TotalHitCountCollectorManager;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.IOUtils;

/**
 * M13 T13.4c: drill-down and drill-sideways. Writes {@code drill_down/}:
 *
 * <ul>
 *   <li>{@code index/} and {@code taxo/}: {@link #DOCS} documents in three segments of 2,600,
 *       1,500 and 400 (deletions in each; the first is wider than {@code DrillSidewaysScorer}'s
 *       2,048-document chunk), built through {@link #config}: a flat {@code Author} (missing from
 *       some documents), a rare flat {@code Rare}, a hierarchical {@code Publish Date}, a
 *       multi-valued {@code Tag}, a hierarchical {@code Color} in its own index field, and a
 *       {@code body} text field to score.
 *   <li>{@code sideways.tsv}: {@code DrillSideways.search(ddq, 10)} for every base query of
 *       {@link #bases} and every drill-down of {@link #drillDowns}, under three modes: {@code seq}
 *       (the {@code DrillSidewaysQuery}), {@code conc} (an executor: one query per dim) and {@code
 *       atonce} ({@code scoreSubDocsAtOnce}); plus {@code count}, the {@code search(ddq,
 *       CollectorManager)} form with a {@code TotalHitCountCollectorManager}. Each line: base,
 *       drill-down, mode, then {@code hits} (total and top docs), {@code dd} (the drill-down
 *       collector: hit count and a digest of its documents), {@code sw} (each sideways dim, in
 *       order: its name, hit count and digest), {@code dims} ({@code facets.getAllDims(3)}, its
 *       {@code null} entries left out) and {@code top} ({@code getTopChildren(3, dim)} for each
 *       dim of the query, then {@code Author} and {@code Tag}).
 *   <li>{@code paging.tsv}: {@code search(after, ddq, n)}, {@code search(ddq, filter, after, n,
 *       null, false)} and topN past {@code maxDoc}.
 *   <li>{@code queries.tsv}: {@code toString} of every {@code DrillDownQuery}, its rewrite, and
 *       the hits of a plain {@code searcher.search(ddq, 20)}; {@code FacetQuery} and {@code
 *       MultiFacetQuery} likewise.
 *   <li>{@code errors.tsv}: refusals and their messages.
 * </ul>
 *
 * A digest is FNV-1a (64-bit) over the global document ids in order. A number is {@code
 * i<int>} or {@code f<float bits, hex>}; a score is its float bits in hex. A string's control
 * characters print as {@code \\u00XX}.
 */
public class GenDrillDown {
  static final int[] SEGMENTS = {2600, 1500, 400};
  static final int DOCS = 4500;

  static FacetsConfig config() {
    FacetsConfig c = new FacetsConfig();
    c.setHierarchical("Publish Date", true);
    c.setMultiValued("Tag", true);
    c.setIndexFieldName("Color", "$color");
    c.setHierarchical("Color", true);
    return c;
  }

  static String body(int i) {
    StringBuilder sb = new StringBuilder();
    for (int k = 0; k < i % 3 + 1; k++) sb.append("alpha ");
    if (i % 4 == 0) sb.append("beta ");
    if (i % 5 < 2) sb.append("gamma ");
    if (i % 6 == 1) sb.append("beta alpha ");
    for (int k = 0; k < i % 7; k++) sb.append("w").append(i % 11).append(' ');
    return sb.toString().trim();
  }

  static Document doc(int i) {
    Document d = new Document();
    d.add(new StringField("id", "d" + i, Field.Store.NO));
    d.add(new StringField("q", "all", Field.Store.NO));
    if (i % 2 == 0) d.add(new StringField("q", "even", Field.Store.NO));
    if (i % 53 == 7) d.add(new StringField("q", "rare", Field.Store.NO));
    if (i % 401 == 3) d.add(new StringField("q", "veryrare", Field.Store.NO));
    if (i % 13 == 4) d.add(new StringField("del", "y", Field.Store.NO));
    d.add(new TextField("body", body(i), Field.Store.NO));
    if (i >= DOCS - 40) return d; // no facets at all
    if (i % 29 != 0) d.add(new FacetField("Author", "a" + (i * 7 % 17)));
    d.add(new FacetField("Rare", "r" + (i % 97)));
    d.add(new FacetField("Publish Date", "" + (2000 + i % 4), "" + (1 + i % 12)));
    d.add(new FacetField("Tag", "t" + (i % 5)));
    d.add(new FacetField("Tag", "t" + (i % 3 + 5)));
    d.add(new FacetField("Color", "c" + (i % 3), "s" + (i % 2)));
    return d;
  }

  static Map<String, Query> bases() {
    Map<String, Query> m = new LinkedHashMap<>();
    m.put("null", null);
    m.put("all", new MatchAllDocsQuery());
    m.put("even", new TermQuery(new Term("q", "even")));
    m.put("rare", new TermQuery(new Term("q", "rare")));
    m.put("veryrare", new TermQuery(new Term("q", "veryrare")));
    m.put("alpha", new TermQuery(new Term("body", "alpha")));
    m.put("gamma", new TermQuery(new Term("body", "gamma")));
    m.put("phrase", new PhraseQuery("body", "alpha", "beta"));
    m.put(
        "bool",
        new BooleanQuery.Builder()
            .add(new TermQuery(new Term("body", "beta")), Occur.SHOULD)
            .add(new TermQuery(new Term("body", "gamma")), Occur.SHOULD)
            .build());
    return m;
  }

  /** A drill-down: each step a dim and a path, or a dim and a custom query ({@code *name}). */
  static Map<String, String[][]> drillDowns() {
    Map<String, String[][]> m = new LinkedHashMap<>();
    m.put("none", new String[][] {});
    m.put("author", new String[][] {{"Author", "a3"}});
    m.put("author_or", new String[][] {{"Author", "a3"}, {"Author", "a5"}});
    m.put("author_tag", new String[][] {{"Author", "a3"}, {"Tag", "t1"}});
    m.put("three", new String[][] {{"Author", "a3"}, {"Tag", "t1"}, {"Publish Date", "2001"}});
    m.put("deep", new String[][] {{"Publish Date", "2001", "3"}});
    m.put("color", new String[][] {{"Color", "c1"}});
    m.put("rare_author", new String[][] {{"Rare", "r5"}, {"Author", "a3"}});
    m.put("rare_author_tag", new String[][] {{"Rare", "r5"}, {"Author", "a3"}, {"Tag", "t6"}});
    m.put("absent_one", new String[][] {{"Author", "zz"}, {"Tag", "t1"}});
    m.put("absent_two", new String[][] {{"Author", "zz"}, {"Tag", "tzz"}});
    m.put("dim_root", new String[][] {{"Tag"}});
    m.put("custom", new String[][] {{"Author", "a3"}, {"Even", "*even"}});
    m.put("phrase_dim", new String[][] {{"Tag", "t2"}, {"Body", "*beta alpha"}});
    m.put(
        "two_phrases",
        new String[][] {{"Body", "*alpha beta"}, {"Body2", "*beta alpha"}, {"Author", "a1"}});
    m.put("mixed_or", new String[][] {{"Tag", "t0"}, {"Tag", "*t6"}, {"Color", "c0", "s1"}});
    return m;
  }

  static Query custom(String spec) {
    String s = spec.substring(1);
    if (s.equals("even")) return new TermQuery(new Term("q", "even"));
    if (s.equals("t6")) return new TermQuery(DrillDownQuery.term("$facets", "Tag", "t6"));
    return new PhraseQuery("body", s.split(" "));
  }

  static DrillDownQuery ddq(FacetsConfig config, Query base, String[][] steps) {
    DrillDownQuery q = base == null ? new DrillDownQuery(config) : new DrillDownQuery(config, base);
    for (String[] step : steps) {
      if (step.length == 2 && step[1].startsWith("*")) {
        q.add(step[0], custom(step[1]));
      } else {
        q.add(step[0], Arrays.copyOfRange(step, 1, step.length));
      }
    }
    return q;
  }

  // ---------------------------------------------------------------------
  // Rendering
  // ---------------------------------------------------------------------

  static String esc(String s) {
    StringBuilder sb = new StringBuilder();
    for (char c : s.toCharArray()) {
      if (c < 0x20 || c == '\\') sb.append(String.format("\\u%04x", (int) c));
      else sb.append(c);
    }
    return sb.toString();
  }

  static String num(Number n) {
    if (n instanceof Float f) return "f" + Integer.toHexString(Float.floatToIntBits(f));
    return "i" + n.intValue();
  }

  static String res(FacetResult r) {
    if (r == null) return "null";
    StringBuilder children = new StringBuilder();
    for (LabelAndValue lv : r.labelValues) {
      if (children.length() > 0) children.append(',');
      children.append(lv.label).append('=').append(num(lv.value)).append('=').append(lv.count);
    }
    return r.dim
        + "|"
        + String.join("/", r.path)
        + "|"
        + num(r.value)
        + "|"
        + r.childCount
        + "|"
        + children;
  }

  static String list(List<FacetResult> l) {
    StringBuilder sb = new StringBuilder();
    for (FacetResult r : l) {
      if (r == null) continue; // Rust's MultiFacets has no null entry
      if (sb.length() > 0) sb.append(';');
      sb.append(res(r));
    }
    return sb.toString();
  }

  static String topDocs(TopDocs t) {
    StringBuilder sb = new StringBuilder();
    sb.append(t.totalHits.value()).append(' ').append(t.totalHits.relation()).append(' ');
    for (int i = 0; i < t.scoreDocs.length; i++) {
      ScoreDoc sd = t.scoreDocs[i];
      if (i > 0) sb.append(',');
      sb.append(sd.doc).append(':').append(Integer.toHexString(Float.floatToIntBits(sd.score)));
    }
    return sb.toString();
  }

  /** Hit count and FNV-1a of the collector's global document ids, ascending. */
  static String digest(FacetsCollector fc) throws Exception {
    List<FacetsCollector.MatchingDocs> mds = new ArrayList<>(fc.getMatchingDocs());
    mds.sort(Comparator.comparingInt(m -> m.context().docBase));
    long h = 0xcbf29ce484222325L;
    long hits = 0;
    long n = 0;
    for (FacetsCollector.MatchingDocs m : mds) {
      hits += m.totalHits();
      DocIdSetIterator it = m.bits().iterator();
      for (int d = it.nextDoc(); d != DocIdSetIterator.NO_MORE_DOCS; d = it.nextDoc()) {
        h ^= m.context().docBase + d;
        h *= 0x100000001b3L;
        n++;
      }
    }
    if (n != hits) throw new AssertionError("bits " + n + " != totalHits " + hits);
    return hits + ":" + Long.toHexString(h);
  }

  static String sideways(DrillSideways.DrillSidewaysResult r) throws Exception {
    if (r.drillSidewaysDims == null) return "-";
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < r.drillSidewaysDims.length; i++) {
      if (i > 0) sb.append(';');
      sb.append(r.drillSidewaysDims[i]).append('=').append(digest(r.drillSidewaysFacetsCollector[i]));
    }
    return sb.toString();
  }

  static String facets(Facets f, String[][] steps) throws Exception {
    if (f == null) return "null\tnull";
    List<String> dims = new ArrayList<>();
    for (String[] s : steps) if (!dims.contains(s[0])) dims.add(s[0]);
    for (String d : new String[] {"Author", "Tag"}) if (!dims.contains(d)) dims.add(d);
    StringBuilder top = new StringBuilder();
    for (String d : dims) {
      if (top.length() > 0) top.append(';');
      try {
        top.append(res(f.getTopChildren(3, d)));
      } catch (IllegalArgumentException e) {
        top.append("err:").append(esc(e.getMessage()));
      }
    }
    String all;
    try {
      all = list(f.getAllDims(3));
    } catch (IllegalArgumentException e) {
      all = "err:" + esc(e.getMessage());
    }
    return all + "\t" + top;
  }

  static String result(DrillSideways.DrillSidewaysResult r, TopDocs hits, String[][] steps)
      throws Exception {
    return "hits\t"
        + (hits == null ? "-" : topDocs(hits))
        + "\tdd\t"
        + (r.drillDownFacetsCollector == null ? "-" : digest(r.drillDownFacetsCollector))
        + "\tsw\t"
        + sideways(r)
        + "\tfacets\t"
        + facets(r.facets, steps);
  }

  static void write(Path p, String s) throws Exception {
    Files.writeString(p, s, StandardCharsets.UTF_8);
  }

  interface Call {
    Object run() throws Exception;
  }

  static void error(StringBuilder sb, String name, Call c) {
    try {
      c.run();
      sb.append(name).append("\tok\n");
    } catch (IllegalArgumentException e) {
      sb.append(name).append('\t').append(esc(e.getMessage())).append('\n');
    } catch (Exception e) {
      throw new AssertionError(name, e);
    }
  }

  public static void main(String[] args) throws Exception {
    Path root = Path.of(args[0]).resolve("drill_down");
    if (Files.exists(root)) IOUtils.rm(root);
    Path indexDir = root.resolve("index");
    Path taxoDir = root.resolve("taxo");
    Files.createDirectories(indexDir);
    Files.createDirectories(taxoDir);
    FacetsConfig config = config();
    try (Directory id = FSDirectory.open(indexDir);
        Directory td = FSDirectory.open(taxoDir)) {
      IndexWriterConfig cfg = new IndexWriterConfig();
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);
      cfg.setMaxBufferedDocs(IndexWriterConfig.DISABLE_AUTO_FLUSH);
      cfg.setRAMBufferSizeMB(256);
      IndexWriter w = new IndexWriter(id, cfg);
      DirectoryTaxonomyWriter tw = new DirectoryTaxonomyWriter(td);
      int i = 0;
      for (int size : SEGMENTS) {
        for (int k = 0; k < size; k++, i++) w.addDocument(config.build(tw, doc(i)));
        w.flush();
      }
      w.deleteDocuments(new Term("del", "y"));
      w.commit();
      w.close();
      tw.close();
    }

    StringBuilder sideways = new StringBuilder();
    StringBuilder paging = new StringBuilder();
    StringBuilder queries = new StringBuilder();
    StringBuilder errors = new StringBuilder();
    ExecutorService executor = Executors.newFixedThreadPool(2);
    try (Directory id = FSDirectory.open(indexDir);
        Directory td = FSDirectory.open(taxoDir);
        DirectoryReader reader = DirectoryReader.open(id);
        DirectoryTaxonomyReader taxo = new DirectoryTaxonomyReader(td)) {
      IndexSearcher searcher = new IndexSearcher(reader);
      if (searcher.getSlices().length != 1) throw new AssertionError("one slice expected");
      if (reader.leaves().size() != SEGMENTS.length) throw new AssertionError("segments");
      DrillSideways seq = new DrillSideways(searcher, config, taxo);
      DrillSideways conc = new DrillSideways(searcher, config, taxo, null, executor);
      DrillSideways atOnce =
          new DrillSideways(searcher, config, taxo) {
            @Override
            protected boolean scoreSubDocsAtOnce() {
              return true;
            }
          };
      Map<String, DrillSideways> modes = new LinkedHashMap<>();
      modes.put("seq", seq);
      modes.put("conc", conc);
      modes.put("atonce", atOnce);

      for (Map.Entry<String, Query> b : bases().entrySet()) {
        for (Map.Entry<String, String[][]> dd : drillDowns().entrySet()) {
          String[][] steps = dd.getValue();
          String pre = b.getKey() + "\t" + dd.getKey() + "\t";
          for (Map.Entry<String, DrillSideways> m : modes.entrySet()) {
            DrillSideways.DrillSidewaysResult r =
                m.getValue().search(ddq(config, b.getValue(), steps), 10);
            sideways.append(pre).append(m.getKey()).append('\t');
            sideways.append(result(r, r.hits, steps)).append('\n');
          }
          DrillSideways.ConcurrentDrillSidewaysResult<Integer> c =
              seq.search(
                  ddq(config, b.getValue(), steps),
                  new TotalHitCountCollectorManager(searcher.getSlices()));
          sideways.append(pre).append("count\t");
          sideways.append("count\t").append(c.collectorResult).append('\t');
          sideways.append(result(c, null, steps).substring("hits\t-\t".length())).append('\n');

          // The query itself.
          DrillDownQuery q = ddq(config, b.getValue(), steps);
          queries
              .append(pre)
              .append(esc(q.toString()))
              .append('\t')
              .append(esc(q.rewrite(searcher).toString()))
              .append('\t')
              .append(topDocs(searcher.search(q, 20)))
              .append('\n');
        }
      }

      // Paging and filters.
      for (String bn : new String[] {"null", "alpha", "bool"}) {
        Query base = bases().get(bn);
        for (String dn : new String[] {"author", "author_tag", "none"}) {
          String[][] steps = drillDowns().get(dn);
          DrillSideways.DrillSidewaysResult first = seq.search(ddq(config, base, steps), 4);
          ScoreDoc last = first.hits.scoreDocs[first.hits.scoreDocs.length - 1];
          DrillSideways.DrillSidewaysResult next = seq.search(last, ddq(config, base, steps), 6);
          paging
              .append(bn + "\t" + dn + "\tafter\t" + last.doc + ":")
              .append(Integer.toHexString(Float.floatToIntBits(last.score)))
              .append('\t')
              .append(result(next, next.hits, steps))
              .append('\n');
          Query filter = new TermQuery(new Term("q", "even"));
          DrillSideways.DrillSidewaysResult f =
              seq.search(ddq(config, base, steps), filter, null, 7, null, false);
          paging
              .append(bn + "\t" + dn + "\tfilter\t-\t")
              .append(result(f, f.hits, steps))
              .append('\n');
          DrillSideways.DrillSidewaysResult fa =
              seq.search(
                  ddq(config, base, steps), filter, new FieldDoc(last.doc, last.score), 7, null,
                  false);
          paging
              .append(bn + "\t" + dn + "\tfilter_after\t-\t")
              .append(result(fa, fa.hits, steps))
              .append('\n');
          DrillSideways.DrillSidewaysResult big = seq.search(ddq(config, base, steps), 100_000);
          paging
              .append(bn + "\t" + dn + "\tbig\t-\t")
              .append(result(big, big.hits, steps))
              .append('\n');
        }
      }

      // FacetQuery and MultiFacetQuery.
      Map<String, Query> fqs = new LinkedHashMap<>();
      fqs.put("facet_author", new FacetQuery(config, "Author", "a3"));
      fqs.put("facet_default", new FacetQuery("Tag", "t1"));
      fqs.put("facet_color", new FacetQuery(config, "Color", "c1", "s0"));
      fqs.put("facet_dim", new FacetQuery(config, "Publish Date"));
      fqs.put("facet_absent", new FacetQuery(config, "Author", "zz"));
      fqs.put(
          "multi_tag",
          new MultiFacetQuery(config, "Tag", new String[] {"t6"}, new String[] {"t1"}));
      fqs.put(
          "multi_default",
          new MultiFacetQuery(
              "Author", new String[] {"a5"}, new String[] {"a3"}, new String[] {"a5"}));
      fqs.put(
          "multi_color",
          new MultiFacetQuery(config, "Color", new String[] {"c2", "s1"}, new String[] {"c0"}));
      fqs.put("multi_none", new MultiFacetQuery(config, "Tag"));
      for (Map.Entry<String, Query> e : fqs.entrySet()) {
        queries
            .append("facetquery\t")
            .append(e.getKey())
            .append('\t')
            .append(esc(e.getValue().toString()))
            .append('\t')
            .append(esc(e.getValue().toString("$facets")))
            .append('\t')
            .append(topDocs(searcher.search(e.getValue(), 20)))
            .append('\n');
      }
      queries
          .append("term\t-\t")
          .append(esc(DrillDownQuery.term("$facets", "Publish Date", "2001", "3").toString()))
          .append('\n');
      // A filter inside a drill-down, and FacetQuery as a drill-down dim.
      DrillDownQuery nested = new DrillDownQuery(config, new TermQuery(new Term("q", "even")));
      nested.add("Tag", new FacetQuery(config, "Tag", "t1"));
      nested.add("Tag", new MultiFacetQuery(config, "Tag", new String[] {"t6"}));
      DrillSideways.DrillSidewaysResult nr = seq.search(nested, 10);
      queries
          .append("nested\t-\t")
          .append(esc(nested.toString()))
          .append('\t')
          .append(result(nr, nr.hits, new String[][] {{"Tag"}}))
          .append('\n');

      // Refusals.
      error(errors, "search_topn_0", () -> seq.search(ddq(config, null, drillDowns().get("author")), 0));
      error(
          errors,
          "search_topn_negative",
          () -> seq.search(ddq(config, null, drillDowns().get("author")), -3));
      error(errors, "add_empty_component", () -> {
        new DrillDownQuery(config).add("Author", "");
        return null;
      });
      error(errors, "facet_query_empty", () -> new FacetQuery(config, "Author", ""));
      error(errors, "multi_facet_query_empty", () -> new MultiFacetQuery(config, "Tag", new String[] {""}));
      error(
          errors,
          "managers_size",
          () ->
              seq.search(
                  ddq(config, null, drillDowns().get("author_tag")),
                  new FacetsCollectorManager(),
                  List.of(new FacetsCollectorManager())));
      error(
          errors,
          "managers_null",
          () ->
              seq.search(
                  ddq(config, null, drillDowns().get("author")),
                  new FacetsCollectorManager(),
                  (List<FacetsCollectorManager>) null));
      error(
          errors,
          "managers_null_no_dims",
          () ->
              seq.search(
                  ddq(config, null, drillDowns().get("none")),
                  new FacetsCollectorManager(),
                  (List<FacetsCollectorManager>) null));
      error(
          errors,
          "managers_ok",
          () ->
              seq.search(
                  ddq(config, null, drillDowns().get("author")),
                  new FacetsCollectorManager(),
                  List.of(new FacetsCollectorManager())));
    } finally {
      executor.shutdown();
    }
    write(root.resolve("sideways.tsv"), sideways.toString());
    write(root.resolve("paging.tsv"), paging.toString());
    write(root.resolve("queries.tsv"), queries.toString());
    write(root.resolve("errors.tsv"), errors.toString());
  }
}
