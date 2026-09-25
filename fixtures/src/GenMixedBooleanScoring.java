import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.LongPoint;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.BoostQuery;
import org.apache.lucene.search.ConstantScoreQuery;
import org.apache.lucene.search.DisjunctionMaxQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.PhraseQuery;
import org.apache.lucene.search.PrefixQuery;
import org.apache.lucene.search.RegexpQuery;
import org.apache.lucene.search.TermInSetQuery;
import org.apache.lucene.search.WildcardQuery;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopScoreDocCollectorManager;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.stream.Stream;

/**
 * Mixed boolean queries -- MUST with SHOULD, MUST_NOT, FILTER, minimum_should_match, boosts,
 * constant_score, dismax and nesting -- over a corpus large enough for block-max pruning to engage,
 * with Lucene's top hits recorded as score bits.
 *
 * <p>The corpus: 24,000 documents in two segments, each 6-14 tokens drawn from a Zipf-like
 * vocabulary {@code w0..w59} ({@code w0} in about half the documents, {@code w59} in a few dozen),
 * so the dense terms span many postings blocks and level-1 skip spans, and their impacts differ
 * per block. Every 37th document of the first segment is deleted, so live docs are in play.
 *
 * <p>Every query is written as an S-expression, which both this generator and the Rust test parse
 * (crates/lucene-search/tests/mixed_boolean_fixtures.rs); keep the two grammars in step:
 *
 * <pre>
 *   (t TERM)                   TermQuery on body
 *   (b MSM CLAUSE...)          BooleanQuery; CLAUSE is (+ Q) MUST, (# Q) FILTER,
 *                              (? Q) SHOULD, (- Q) MUST_NOT
 *   (boost F Q)                BoostQuery
 *   (const Q)                  ConstantScoreQuery (score 1)
 *   (dismax TIE Q...)          DisjunctionMaxQuery
 *   (p TERM...)                PhraseQuery on body
 *   (ps SLOP TERM...)          sloppy PhraseQuery
 *   (pre PREFIX)               PrefixQuery on body
 *   (wc PATTERN)               WildcardQuery on body
 *   (re PATTERN)               RegexpQuery on body
 *   (ts TERM...)               TermInSetQuery on body
 *   (r MIN MAX)                LongPoint.newRangeQuery on n (each document's id)
 * </pre>
 *
 * For each query the manifest records the top {@code N} under a total-hits threshold of 100 (so
 * the collector publishes a minimum competitive score early and the scorers prune), and the top
 * {@code N} with an exact count (no pruning), as {@code doc:floatBits} pairs in global doc ids.
 */
public class GenMixedBooleanScoring {
  static final int DOCS_PER_SEGMENT = 12_000;
  static final int VOCAB = 60;
  static final int TOP_N = 10;
  static final int THRESHOLD = 100;

  static final String[] QUERIES = {
    "(b 0 (+ (t w0)) (? (t w1)))",
    "(b 0 (+ (t w1)) (? (t w0)) (? (t w2)))",
    "(b 0 (+ (t w0)) (- (t w1)))",
    "(b 0 (? (t w0)) (? (t w3)) (- (t w2)))",
    "(b 0 (+ (t w0)) (+ (t w1)) (? (t w2)) (- (t w9)))",
    "(b 2 (? (t w0)) (? (t w1)) (? (t w2)) (? (t w5)))",
    "(b 3 (? (t w0)) (? (t w1)) (? (t w2)) (? (t w3)) (? (t w4)))",
    "(b 0 (+ (b 0 (? (t w0)) (? (t w4)))) (+ (t w2)))",
    "(b 0 (+ (b 1 (? (t w1)) (? (t w2)) (? (t w3)))) (# (t w0)))",
    "(boost 2.5 (t w1))",
    "(b 0 (? (boost 3 (t w0))) (? (t w7)))",
    "(b 0 (+ (t w0)) (? (boost 0.5 (t w1))) (? (t w2)) (- (t w3)))",
    "(const (t w1))",
    "(boost 2 (const (t w1)))",
    "(b 0 (? (const (t w1))) (? (t w9)))",
    "(b 0 (+ (const (t w0))) (? (t w12)))",
    "(b 0 (# (t w0)) (? (t w5)) (? (t w6)))",
    "(b 1 (# (t w0)) (? (t w5)) (? (t w6)))",
    "(b 0 (# (t w0)) (- (t w1)))",
    "(b 0 (# (t w0)) (# (t w1)) (- (t w2)))",
    "(dismax 0.1 (t w2) (t w3) (t w8))",
    "(dismax 0 (t w0) (t w1))",
    "(b 0 (+ (dismax 0 (t w1) (t w2))) (? (t w3)))",
    "(b 0 (+ (t w40)) (? (t w0)) (? (t w1)))",
    "(b 0 (+ (t w0)) (+ (t w55)) (? (t w1)))",
    "(b 0 (? (t w30)) (? (t w31)) (? (t w32)) (- (t w0)))",
    "(b 1 (? (t w0)) (? (t w1)) (? (t nosuchterm)))",
    "(b 2 (? (t w0)) (? (t w1)) (? (t nosuchterm)))",
    "(b 0 (+ (t w0)) (? (b 0 (+ (t w1)) (? (t w2)))) (? (boost 1.5 (t w3))))",
    "(b 0 (+ (b 0 (+ (t w1)) (- (t w2)))) (? (t w0)) (- (t w4)))",
    // Shapes added after R1's review, one per path no earlier query reached:
    // several filters with several optional clauses (the filtered-MaxScore
    // switch, whose filter legs arrive out of step with each other), ...
    "(b 0 (# (t w3)) (# (t w1)) (? (t w5)) (? (t w6)))",
    "(b 0 (# (t w20)) (# (t w1)) (? (t w0)) (? (t w2)))",
    "(b 0 (# (t w8)) (# (t w9)) (? (t w0)) (? (t w1)))",
    "(b 0 (# (t w12)) (# (t w13)) (# (t w14)) (? (t w2)) (? (t w3)))",
    // Filters of similar density, so the non-lead one often overshoots the
    // lead's next document before the switch to MaxScore (review finding).
    "(b 0 (# (t w2)) (# (t w3)) (? (t w0)) (? (t w1)))",
    "(b 0 (# (t w4)) (# (t w5)) (? (t w0)) (? (t w6)))",
    "(b 0 (# (t w5)) (# (t w6)) (? (t w1)) (? (t w2)))",
    "(b 0 (# (t w6)) (# (t w7)) (? (t w0)) (? (t w3)))",
    // ... a two-phase clause (a WAND with minimum_should_match 2) in a
    // conjunction, several MUST_NOTs, a conjunction nested in a disjunction,
    "(b 0 (+ (b 2 (? (t w1)) (? (t w2)) (? (t w3)))) (+ (t w0)))",
    "(b 0 (+ (t w0)) (- (t w1)) (- (t w2)))",
    "(b 0 (? (b 0 (+ (t w1)) (+ (t w2)))) (? (t w3)))",
    // ... required-plus-optional led by rare optional clauses, one and many,
    "(b 0 (+ (t w0)) (? (t w1)) (? (t w2)) (? (t w3)))",
    "(b 0 (+ (t w0)) (? (t w40)) (? (t w41)))",
    "(b 0 (+ (t w0)) (? (t w40)))",
    "(b 0 (+ (t w1)) (+ (t w2)) (? (t w3)) (? (t w4)))",
    // ... a three-level boost chain (the association of its product),
    "(boost 2 (boost 3 (boost 0.7 (t w1))))",
    // ... and nested non-term clauses in every position.
    "(b 0 (+ (dismax 0.5 (t w1) (t w4))) (+ (t w2)) (+ (t w3)))",
    "(b 1 (# (t w0)) (# (t w2)) (? (t w1)) (? (t w5)))",
    "(b 0 (# (t w1)))",
    "(b 0 (+ (t w1)) (# (t w0)) (- (t w2)))",
    "(b 0 (+ (boost 1.5 (b 0 (? (t w1)) (? (t w2))))) (- (t w3)))",
    "(b 0 (# (t w0)) (? (b 0 (+ (t w1)) (+ (t w2)))) (? (t w3)))",
    "(dismax 0.2 (b 0 (+ (t w1)) (+ (t w2))) (t w3))",
    "(b 0 (+ (const (b 0 (? (t w1)) (? (t w2))))) (? (t w5)))",
    // minimum_should_match with fewer SHOULD clauses than it: nothing matches,
    // even though a MUST clause alone would (BooleanQuery.rewrite only
    // unwraps a lone MUST when minimum_should_match is 0).
    "(b 1 (+ (t w1)))",
    "(b 2 (+ (t w1)) (? (t w2)))",
    // A top-level dismax of terms runs a window at a time (`DisMaxBulk`):
    // dense and sparse legs, a boost, a constant-scored leg, an absent term.
    "(dismax 0.3 (t w0) (t w1))",
    "(boost 2 (dismax 0.5 (t w0) (t w3) (t w9)))",
    "(dismax 0.1 (t w1) (const (t w2)))",
    "(dismax 0.7 (t w0) (t nosuchterm))",
    "(dismax 0.25 (t w5) (t w6) (t w7) (t w8) (t w9))",
    // `MUST` + `SHOULD` with a minimum: `ConjunctionScorer(req, opt)` in
    // Lucene, a block-max conjunction of the same two scorers here.
    "(b 1 (+ (t w0)) (? (t w1)) (? (t w2)))",
    "(b 2 (+ (t w1)) (? (t w0)) (? (t w3)) (? (t w5)))",
    "(b 1 (+ (t w0)) (+ (t w2)) (? (t w1)) (? (t w4)) (- (t w6)))",
    "(b 1 (# (t w0)) (+ (t w3)) (? (boost 3 (t w1))) (? (t w9)))",
    // Dispatch branches over clauses that are not all terms: a filtered
    // MaxScore over a nested boolean, a block-max conjunction with a filter,
    // and a lone non-term FILTER (scored as 0).
    "(b 1 (# (t w0)) (? (b 0 (+ (t w1)) (+ (t w2)))) (? (t w3)))",
    "(b 0 (+ (b 0 (? (t w1)) (? (t w2)))) (+ (t w3)) (# (t w4)))",
    "(b 0 (# (b 0 (? (t w1)) (? (t w2)))))",
    // Phrases inside the tree, two-phase (`PhraseScorer`): required, optional,
    // excluded, in a dismax, constant-scored, sloppy, with a repeated term,
    // under a minimum_should_match.
    "(b 0 (+ (p w0 w1)) (? (t w2)))",
    "(b 0 (? (p w0 w1)) (? (p w1 w2)) (? (t w5)))",
    "(b 0 (+ (t w0)) (+ (p w1 w0)))",
    "(b 0 (+ (t w3)) (- (p w0 w1)))",
    "(dismax 0.2 (p w0 w1) (t w4))",
    "(b 1 (+ (t w0)) (? (ps 2 w1 w3)) (? (t w6)))",
    "(b 0 (+ (const (p w2 w0))) (? (t w1)))",
    "(b 0 (+ (ps 1 w0 w0)) (? (t w1)))",
    "(b 2 (? (p w0 w1)) (? (t w2)) (? (t w3)))",
    "(b 0 (# (p w0 w2)) (? (t w1)) (? (t w4)))",
    // Sloppy phrases with three terms and repeated terms, alone and nested.
    "(ps 1 w0 w1 w0)",
    "(ps 2 w0 w0 w1)",
    "(ps 2 w3 w1 w2)",
    "(p w0 w0)",
    "(p w0 w1 w0)",
    "(b 0 (? (ps 2 w0 w1 w2)) (? (t w3)))",
    "(b 0 (? (ps 1 w1 w1)) (? (p w2 w2 w0)))",
    // Phrases over pulsed singletons (docFreq 1 in each segment).
    "(p solo pair)",
    "(b 0 (? (p solo pair)) (? (t w0)))",
    "(b 0 (+ (t w0)) (+ (p pair w0)))",
    "(b 0 (? (ps 2 pair solo)) (? (t w1)))",
    // A root phrase under a boost runs `PhraseScorer` itself, which then
    // receives the collector's threshold (the `maxFreq` check).
    "(boost 2 (p w0 w1))",
    "(boost 0.5 (p w1 w0 w2))",
    // The multi-term family: up to 16 terms a constant-scored disjunction,
    // past that the blended rewrite (16 iterators and a bitset).
    "(pre w1)",
    "(b 0 (+ (t w2)) (# (pre w1)))",
    "(b 0 (? (pre w)) (? (t w3)))",
    "(b 0 (+ (t w0)) (- (wc w?5)))",
    "(b 0 (? (re w[1-3][0-9])) (? (t w1)))",
    "(b 0 (+ (ts w5 w9 w14 w40 nosuchterm)) (? (t w0)))",
    "(ts w1 w2)",
    "(dismax 0.3 (pre w4) (t w2))",
    "(boost 3 (wc w1*))",
    // Points ranges: a filter, an exclusion, alone, scored constant, spanning
    // both segments, and empty.
    "(b 0 (+ (t w0)) (# (r 1000 8000)))",
    "(b 0 (+ (t w1)) (- (r 0 11999)))",
    "(r 11990 12010)",
    "(b 0 (? (r 500 600)) (? (t w2)))",
    "(b 0 (+ (t w3)) (# (r 30000 40000)))",
    // Ranges the reader-level rewrite resolves: every value (a match-all
    // filter the boolean then drops), inside a constant score, boosted, and
    // past every value.
    "(b 0 (+ (t w2)) (# (r -5 99999)))",
    "(b 0 (+ (t w4)) (# (const (r -5 99999))) (? (t w1)))",
    "(boost 2 (r 100 200))",
    "(b 0 (? (r 50000 60000)) (? (t w5)))",
  };

  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]).resolve("mixed_boolean_scoring_index");
    if (Files.exists(out)) {
      try (Stream<Path> s = Files.walk(out)) {
        s.sorted(Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
      }
    }
    Files.createDirectories(out);

    StringBuilder m = new StringBuilder();
    try (Directory dir = FSDirectory.open(out)) {
      IndexWriterConfig cfg = new IndexWriterConfig(new StandardAnalyzer());
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);
      cfg.setRAMBufferSizeMB(256);
      cfg.setMaxBufferedDocs(IndexWriterConfig.DISABLE_AUTO_FLUSH);
      Random random = new Random(20260925L);
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        for (int seg = 0; seg < 2; seg++) {
          for (int i = 0; i < DOCS_PER_SEGMENT; i++) {
            int id = seg * DOCS_PER_SEGMENT + i;
            Document doc = new Document();
            doc.add(new StringField("id", Integer.toString(id), Field.Store.NO));
            String body = body(random);
            if (i == 5) {
              // `solo` and `pair` occur in one document per segment: pulsed
              // singletons (docFreq 1, no .doc stream), inside phrases below.
              body = "solo pair w0 w1 solo " + body;
            }
            doc.add(new TextField("body", body, Field.Store.NO));
            doc.add(new LongPoint("n", id));
            w.addDocument(doc);
          }
          w.commit();
        }
        for (int id = 0; id < DOCS_PER_SEGMENT; id += 37) {
          w.deleteDocuments(new Term("id", Integer.toString(id)));
        }
        w.commit();
      }

      SegmentInfos sis = SegmentInfos.readLatestCommit(dir);
      if (sis.size() != 2) {
        throw new AssertionError("expected exactly two segments, got " + sis.size());
      }
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        IndexSearcher searcher = new IndexSearcher(reader);
        searcher.setQueryCache(null);
        m.append("top_n=").append(TOP_N).append('\n');
        m.append("threshold=").append(THRESHOLD).append('\n');
        m.append("num_docs=").append(reader.numDocs()).append('\n');
        m.append("query_count=").append(QUERIES.length).append('\n');
        for (int i = 0; i < QUERIES.length; i++) {
          Query q = parse(new Tokens(QUERIES[i]));
          m.append("query.").append(i).append('=').append(QUERIES[i]).append('\n');
          TopDocs pruned = searcher.search(q, new TopScoreDocCollectorManager(TOP_N, THRESHOLD));
          m.append("query.").append(i).append(".pruned=").append(hits(pruned)).append('\n');
          m.append("query.").append(i).append(".pruned.relation=")
              .append(pruned.totalHits.relation() == TotalHits.Relation.EQUAL_TO ? "eq" : "gte")
              .append('\n');
          TopDocs exact =
              searcher.search(q, new TopScoreDocCollectorManager(TOP_N, Integer.MAX_VALUE));
          if (exact.totalHits.relation() != TotalHits.Relation.EQUAL_TO) {
            throw new AssertionError("exact count expected for " + QUERIES[i]);
          }
          m.append("query.").append(i).append(".exact=").append(hits(exact)).append('\n');
          m.append("query.").append(i).append(".total=").append(exact.totalHits.value()).append('\n');
        }
      }
    }
    Files.writeString(out.resolve("manifest.properties"), m.toString());
    System.out.println("wrote " + out);
  }

  /** 6-14 tokens; token k is drawn with probability proportional to 1 / (k + 1). */
  static String body(Random random) {
    double norm = 0;
    for (int k = 0; k < VOCAB; k++) {
      norm += 1.0 / (k + 1);
    }
    int len = 6 + random.nextInt(9);
    StringBuilder b = new StringBuilder();
    for (int t = 0; t < len; t++) {
      double u = random.nextDouble() * norm;
      int k = 0;
      double acc = 1.0;
      while (acc < u && k < VOCAB - 1) {
        k++;
        acc += 1.0 / (k + 1);
      }
      if (t > 0) {
        b.append(' ');
      }
      b.append('w').append(k);
    }
    return b.toString();
  }

  static String hits(TopDocs td) {
    StringBuilder b = new StringBuilder();
    for (ScoreDoc sd : td.scoreDocs) {
      if (b.length() > 0) {
        b.append(',');
      }
      b.append(sd.doc).append(':').append(Float.floatToIntBits(sd.score));
    }
    return b.toString();
  }

  static final class Tokens {
    final List<String> toks = new ArrayList<>();
    int at;

    Tokens(String s) {
      for (String t : s.replace("(", " ( ").replace(")", " ) ").trim().split("\\s+")) {
        toks.add(t);
      }
    }

    String next() {
      return toks.get(at++);
    }

    String peek() {
      return toks.get(at);
    }

    void expect(String t) {
      String got = next();
      if (!got.equals(t)) {
        throw new IllegalArgumentException("expected " + t + ", got " + got);
      }
    }
  }

  static Query parse(Tokens t) {
    t.expect("(");
    String op = t.next();
    Query q;
    switch (op) {
      case "t" -> q = new TermQuery(new Term("body", t.next()));
      case "r" -> q = LongPoint.newRangeQuery("n", Long.parseLong(t.next()), Long.parseLong(t.next()));
      case "pre" -> q = new PrefixQuery(new Term("body", t.next()));
      case "wc" -> q = new WildcardQuery(new Term("body", t.next()));
      case "re" -> q = new RegexpQuery(new Term("body", t.next()));
      case "ts" -> {
        List<BytesRef> terms = new ArrayList<>();
        while (!t.peek().equals(")")) {
          terms.add(new BytesRef(t.next()));
        }
        q = new TermInSetQuery("body", terms);
      }
      case "p", "ps" -> {
        int slop = op.equals("ps") ? Integer.parseInt(t.next()) : 0;
        List<String> words = new ArrayList<>();
        while (!t.peek().equals(")")) {
          words.add(t.next());
        }
        q = new PhraseQuery(slop, "body", words.toArray(new String[0]));
      }
      case "boost" -> {
        float f = Float.parseFloat(t.next());
        q = new BoostQuery(parse(t), f);
      }
      case "const" -> q = new ConstantScoreQuery(parse(t));
      case "dismax" -> {
        float tie = Float.parseFloat(t.next());
        List<Query> ds = new ArrayList<>();
        while (t.peek().equals("(")) {
          ds.add(parse(t));
        }
        q = new DisjunctionMaxQuery(ds, tie);
      }
      case "b" -> {
        BooleanQuery.Builder b = new BooleanQuery.Builder();
        b.setMinimumNumberShouldMatch(Integer.parseInt(t.next()));
        while (t.peek().equals("(")) {
          t.expect("(");
          BooleanClause.Occur occur =
              switch (t.next()) {
                case "+" -> BooleanClause.Occur.MUST;
                case "#" -> BooleanClause.Occur.FILTER;
                case "?" -> BooleanClause.Occur.SHOULD;
                case "-" -> BooleanClause.Occur.MUST_NOT;
                default -> throw new IllegalArgumentException("bad occur");
              };
          b.add(parse(t), occur);
          t.expect(")");
        }
        q = b.build();
      }
      default -> throw new IllegalArgumentException("unknown op " + op);
    }
    t.expect(")");
    return q;
  }
}
