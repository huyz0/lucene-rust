import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.*;
import org.apache.lucene.search.BooleanClause.Occur;
import org.apache.lucene.search.similarities.BooleanSimilarity;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.tests.search.DisablingBulkScorerQuery;
import org.apache.lucene.tests.search.ScorerIndexSearcher;

/** Actual ScorerIndexSearcher and DisablingBulkScorerQuery over three committed leaves. */
public class GenScorerIndexSearcher {
  static final String[] TEXT = {"a b a", "b c", "a c b", "a a c", "d b", "a b c", "a c", "b d", "c d", "a a b", "a d c", "a b d"};
  static final StringBuilder OUT = new StringBuilder();
  static void row(Object... cells) {
    for (int i = 0; i < cells.length; i++) { if (i != 0) OUT.append('\t'); OUT.append(cells[i]); }
    OUT.append('\n');
  }
  static Analyzer analyzer() { return new StandardAnalyzer(); }
  static void write(ByteBuffersDirectory directory, int repeats) throws Exception {
    try (Analyzer a = analyzer(); IndexWriter writer = new IndexWriter(directory,
        new IndexWriterConfig(a).setMergePolicy(NoMergePolicy.INSTANCE))) {
      for (int segment = 0; segment < 3; segment++) {
        for (int rep = 0; rep < repeats; rep++) for (int i = segment * 4; i < segment * 4 + 4; i++) {
          Document doc = new Document();
          doc.add(new StringField("id", Integer.toString(i), Field.Store.NO));
          doc.add(new TextField("body", TEXT[i], Field.Store.NO));
          writer.addDocument(doc);
        }
        writer.commit();
      }
      for (int id : new int[] {1, 6, 10}) writer.deleteDocuments(new Term("id", Integer.toString(id)));
      writer.commit();
    }
  }
  static Query term(String value) { return new TermQuery(new Term("body", value)); }
  static Query bool(Query a, Occur oa, Query b, Occur ob) {
    return new BooleanQuery.Builder().add(a, oa).add(b, ob).build();
  }
  static Query[] queries() {
    Query phrase = new PhraseQuery("body", "a", "b");
    return new Query[] {term("a"), bool(term("a"), Occur.SHOULD, term("b"), Occur.SHOULD),
      bool(term("a"), Occur.MUST, term("b"), Occur.MUST),
      bool(term("a"), Occur.MUST, term("b"), Occur.SHOULD),
      bool(term("a"), Occur.MUST, term("d"), Occur.MUST_NOT), phrase,
      bool(term("a"), Occur.MUST, phrase, Occur.FILTER), new ConstantScoreQuery(phrase),
      new MatchAllDocsQuery(), term("absent")};
  }
  static final String[] THRESHOLD_NAMES = {"direct", "dismax", "dismax_tie", "dismax_singleton", "dismax_singleton_tie", "blended", "single_phrase", "required_filter", "required_exclusion"};
  static Query[] thresholdQueries() {
    return new Query[] {term("a"),
      new DisjunctionMaxQuery(List.of(term("a"), term("b")), 0f),
      new DisjunctionMaxQuery(List.of(term("a"), term("b")), 0.5f),
      new DisjunctionMaxQuery(List.of(term("a"), term("absent")), 0f),
      new DisjunctionMaxQuery(List.of(term("a"), term("absent")), 0.5f),
      new BlendedTermQuery.Builder().add(new Term("body", "a")).add(new Term("body", "b"))
        .setRewriteMethod(new BlendedTermQuery.DisjunctionMaxRewrite(0f)).build(),
      new PhraseQuery("body", "a"),
      bool(term("a"), Occur.MUST, term("b"), Occur.FILTER),
      bool(term("a"), Occur.MUST, term("d"), Occur.MUST_NOT)};
  }
  static final String[] CONSTANT_THRESHOLD_NAMES = {"constant_term", "boosted_constant_term", "nested_boosted_constant_term", "constant_required_filter", "constant_required_exclusion", "constant_disjunction", "constant_dismax", "constant_dismax_tie", "constant_dismax_singleton", "constant_dismax_singleton_tie", "constant_dismax_mixed"};
  static Query[] constantThresholdQueries() {
    return new Query[] {new ConstantScoreQuery(term("a")),
      new BoostQuery(new ConstantScoreQuery(term("a")), 2f),
      new BoostQuery(new ConstantScoreQuery(new BoostQuery(new ConstantScoreQuery(term("a")), 3f)), 2f),
      bool(new ConstantScoreQuery(term("a")), Occur.MUST, term("b"), Occur.FILTER),
      bool(new ConstantScoreQuery(term("a")), Occur.MUST, term("d"), Occur.MUST_NOT),
      bool(new ConstantScoreQuery(term("a")), Occur.SHOULD, new ConstantScoreQuery(term("b")), Occur.SHOULD),
      new DisjunctionMaxQuery(List.of(new ConstantScoreQuery(term("a")), new ConstantScoreQuery(term("b"))), 0f),
      new DisjunctionMaxQuery(List.of(new ConstantScoreQuery(term("a")), new ConstantScoreQuery(term("b"))), 0.5f),
      new DisjunctionMaxQuery(List.of(new ConstantScoreQuery(term("a")), term("absent")), 0f),
      new DisjunctionMaxQuery(List.of(new ConstantScoreQuery(term("a")), term("absent")), 0.5f),
      new DisjunctionMaxQuery(List.of(new ConstantScoreQuery(term("a")), term("b")), 0f)};
  }
  static class Searcher extends ScorerIndexSearcher {
    Searcher(DirectoryReader reader, ExecutorService pool) { super(reader, pool); }
    @Override protected LeafSlice[] slices(List<LeafReaderContext> leaves) {
      return IndexSearcher.slices(leaves, 1, 1, false);
    }
    void selected(Query query, Hits collector) throws IOException {
      Weight w = createWeight(rewrite(query), collector.scoreMode(), 1);
      for (int index : new int[] {0, 2}) {
        LeafReaderContext leaf = getIndexReader().leaves().get(index);
        searchLeaf(leaf, 0, DocIdSetIterator.NO_MORE_DOCS, w, collector);
      }
    }
  }
  record Hit(int doc, int bits) {}
  static class Hits implements Collector {
    final ScoreMode mode;
    final float minimumScore;
    final List<Hit> hits = new ArrayList<>();
    Hits(ScoreMode mode) { this(mode, Float.NEGATIVE_INFINITY); }
    Hits(ScoreMode mode, boolean infinity) { this(mode, infinity ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY); }
    Hits(ScoreMode mode, float minimumScore) { this.mode = mode; this.minimumScore = minimumScore; }
    @Override public ScoreMode scoreMode() { return mode; }
    @Override public LeafCollector getLeafCollector(LeafReaderContext ctx) {
      return new LeafCollector() {
        Scorable scorer;
        @Override public void setScorer(Scorable scorer) throws IOException { this.scorer = scorer; if (minimumScore != Float.NEGATIVE_INFINITY) scorer.setMinCompetitiveScore(minimumScore); }
        @Override public void collect(int doc) throws IOException {
          hits.add(new Hit(ctx.docBase + doc, mode.needsScores() ? Float.floatToRawIntBits(scorer.score()) : 0));
          if (minimumScore != Float.NEGATIVE_INFINITY) scorer.setMinCompetitiveScore(minimumScore);
        }
      };
    }
    String text() { return hitsText(hits); }
  }
  static String hitsText(List<Hit> hits) {
    StringBuilder out = new StringBuilder();
    for (Hit hit : hits) { if (!out.isEmpty()) out.append(','); out.append(hit.doc).append(':').append(String.format("%08x", hit.bits)); }
    return out.toString();
  }
  static String topText(TopDocs top) {
    List<Hit> hits = new ArrayList<>();
    for (ScoreDoc hit : top.scoreDocs) hits.add(new Hit(hit.doc, Float.floatToRawIntBits(hit.score)));
    return top.totalHits.value() + ":" + top.totalHits.relation() + ":" + hitsText(hits);
  }
  static class Manager implements CollectorManager<Hits, List<Hit>> {
    @Override public Hits newCollector() { return new Hits(ScoreMode.COMPLETE); }
    @Override public List<Hit> reduce(Collection<Hits> collectors) {
      List<Hit> hits = new ArrayList<>(); for (Hits c : collectors) hits.addAll(c.hits);
      hits.sort(Comparator.comparingInt(Hit::doc)); return hits;
    }
  }
  /** A specialized bulk scorer must not be requested by either wrapper. */
  static Query trap(Query inner, AtomicInteger gets, AtomicInteger bulks, AtomicInteger tops) {
    return new Query() {
      @Override public Weight createWeight(IndexSearcher searcher, ScoreMode mode, float boost) throws IOException {
        Weight weight = searcher.createWeight(inner, mode, boost);
        return new FilterWeight(weight) {
          @Override public ScorerSupplier scorerSupplier(LeafReaderContext ctx) throws IOException {
            ScorerSupplier source = weight.scorerSupplier(ctx);
            if (source == null) return null;
            return new ScorerSupplier() {
              @Override public long cost() { return source.cost(); }
              @Override public void setTopLevelScoringClause() throws IOException { tops.incrementAndGet(); source.setTopLevelScoringClause(); }
              @Override public Scorer get(long leadCost) throws IOException { gets.incrementAndGet(); return source.get(leadCost); }
              @Override public BulkScorer bulkScorer() { bulks.incrementAndGet(); throw new AssertionError("specialized bulk requested"); }
            };
          }
        };
      }
      @Override public String toString(String field) { return "bulk-trap(" + inner + ")"; }
      @Override public void visit(QueryVisitor visitor) { visitor.visitLeaf(this); }
      @Override public boolean equals(Object other) { return this == other; }
      @Override public int hashCode() { return System.identityHashCode(this); }
    };
  }
  static void fixture(DirectoryReader reader, ExecutorService pool) throws Exception {
    row("reader", reader.leaves().size(), reader.maxDoc(), reader.numDocs());
    for (String mode : new String[] {"scorer_searcher", "disabling_query"}) {
      for (String sim : new String[] {"bm25", "boolean"}) {
        Searcher forced = new Searcher(reader, pool);
        IndexSearcher plain = new IndexSearcher(reader, pool);
        if (sim.equals("boolean")) { forced.setSimilarity(new BooleanSimilarity()); plain.setSimilarity(new BooleanSimilarity()); }
        IndexSearcher searcher = mode.equals("scorer_searcher") ? forced : plain;
        Query[] queries = queries();
        for (int q = 0; q < queries.length; q++) {
          Query query = mode.equals("disabling_query") ? new DisablingBulkScorerQuery(queries[q]) : queries[q];
          try {
            for (ScoreMode scoreMode : new ScoreMode[] {ScoreMode.COMPLETE, ScoreMode.COMPLETE_NO_SCORES, ScoreMode.TOP_DOCS, ScoreMode.TOP_DOCS_WITH_SCORES}) {
              Hits hits = new Hits(scoreMode); searcher.search(query, hits);
              row("hits", mode, sim, q, scoreMode, hits.text());
            }
            TopDocs top = searcher.search(query, 3);
            row("top", mode, sim, q, topText(top));
            TopDocs after = top.scoreDocs.length == 0 ? top : searcher.searchAfter(top.scoreDocs[0], query, 3);
            row("after", mode, sim, q, topText(after));
            row("count", mode, sim, q, searcher.count(query));
            row("manager", mode, sim, q, hitsText(searcher.search(query, new Manager())));
            if (mode.equals("scorer_searcher")) {
              Hits slice = new Hits(ScoreMode.COMPLETE); forced.selected(query, slice);
              row("slice", mode, sim, q, slice.text());
            }
          } catch (NullPointerException expected) {
            if (!mode.equals("disabling_query") || q != queries.length - 1) throw expected;
            row("error", mode, sim, q, "NullPointerException:null-supplier");
          }
        }
        AtomicInteger gets = new AtomicInteger(), bulks = new AtomicInteger(), tops = new AtomicInteger();
        Query query = trap(term("a"), gets, bulks, tops);
        if (mode.equals("disabling_query")) query = new DisablingBulkScorerQuery(query);
        Hits hits = new Hits(ScoreMode.COMPLETE); searcher.search(query, hits);
        row("trap", mode, sim, gets.get(), bulks.get(), tops.get(), hits.text());
        Query[] thresholdQueries = thresholdQueries();
        for (int t = 0; t < thresholdQueries.length; t++) {
          Query thresholdQuery = mode.equals("disabling_query") ? new DisablingBulkScorerQuery(thresholdQueries[t]) : thresholdQueries[t];
          Hits threshold = new Hits(ScoreMode.TOP_SCORES, true); searcher.search(thresholdQuery, threshold);
          row("threshold", mode, sim, THRESHOLD_NAMES[t], threshold.text());
        }
        // ConstantScoreScorer responds to scoreMode, independently of the
        // supplier top-level notification that real TermScorer requires.
        Query[] constantQueries = constantThresholdQueries();
        String[] thresholdLabels = {"one", "two", "infinity"};
        float[] minimumScores = {1f, 2f, Float.POSITIVE_INFINITY};
        for (int q = 0; q < constantQueries.length; q++) {
          Query constantQuery = mode.equals("disabling_query") ? new DisablingBulkScorerQuery(constantQueries[q]) : constantQueries[q];
          for (ScoreMode scoreMode : ScoreMode.values()) for (int t = 0; t < minimumScores.length; t++) {
            Hits threshold = new Hits(scoreMode, minimumScores[t]); searcher.search(constantQuery, threshold);
            row("constant_threshold", mode, sim, CONSTANT_THRESHOLD_NAMES[q], scoreMode, thresholdLabels[t], threshold.text());
          }
        }
        List<Hit> ordinaryThreshold = new ArrayList<>();
        Weight ordinaryWeight = plain.createWeight(plain.rewrite(term("a")), ScoreMode.TOP_DOCS_WITH_SCORES, 1f);
        for (LeafReaderContext leaf : reader.leaves()) {
          ScorerSupplier supplier = ordinaryWeight.scorerSupplier(leaf);
          if (supplier == null) continue;
          Scorer scorer = supplier.get(Long.MAX_VALUE);
          scorer.setMinCompetitiveScore(Float.POSITIVE_INFINITY);
          DocIdSetIterator iterator = scorer.iterator();
          for (int doc = iterator.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = iterator.nextDoc()) {
            if (leaf.reader().getLiveDocs() == null || leaf.reader().getLiveDocs().get(doc)) {
              ordinaryThreshold.add(new Hit(leaf.docBase + doc, Float.floatToRawIntBits(scorer.score())));
              scorer.setMinCompetitiveScore(Float.POSITIVE_INFINITY);
            }
          }
        }
        row("ordinary_threshold", mode, sim, "TOP_DOCS_WITH_SCORES", hitsText(ordinaryThreshold));
        searcher.setTimeout(() -> true); Hits stopped = new Hits(ScoreMode.COMPLETE); searcher.search(query, stopped);
        row("timeout", mode, sim, stopped.hits.size(), searcher.timedOut());
      }
    }
  }
  public static void main(String[] args) throws Exception {
    try (ByteBuffersDirectory directory = new ByteBuffersDirectory()) {
      write(directory, 1);
      try (DirectoryReader reader = DirectoryReader.open(directory); ExecutorService pool = Executors.newFixedThreadPool(2)) { fixture(reader, pool); }
    }
    Path out = Path.of(args[0]); Files.createDirectories(out);
    Files.writeString(out.resolve("scorer_index_searcher.tsv"), OUT.toString());
  }
}
