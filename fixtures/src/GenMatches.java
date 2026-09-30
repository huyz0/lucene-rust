import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.LongPoint;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.BoostQuery;
import org.apache.lucene.search.ConstantScoreQuery;
import org.apache.lucene.search.DisjunctionMaxQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.Matches;
import org.apache.lucene.search.MatchesIterator;
import org.apache.lucene.search.MatchesUtils;
import org.apache.lucene.search.NamedMatches;
import org.apache.lucene.search.PhraseQuery;
import org.apache.lucene.search.PrefixQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.RegexpQuery;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.TermInSetQuery;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.Weight;
import org.apache.lucene.search.WildcardQuery;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.TreeSet;
import java.util.stream.Stream;

/**
 * {@code Weight.matches(context, doc)} -- the Matches API -- recorded from Lucene for
 * crates/lucene-search/tests/matches_fixtures.rs.
 *
 * <p>Two segments (the first with deletions) with a positions-and-offsets text field {@code body},
 * a positions-only field {@code plain}, a freqs-only field {@code tag}, and a long point {@code r}.
 * For each query and every document (deleted ones too: matches ignore deletions), the matches as
 * {@code m.Q.DOC=field|sp:ep:so:eo:QueryClass,...;field|...} with {@code noterms} for {@code
 * MATCH_WITH_NO_TERMS}; and, for a boolean of named clauses, {@code named.DOC=} the names {@code
 * NamedMatches.findNamedMatches} finds.
 */
public class GenMatches {
  static final int[] SEGMENTS = {150, 150};

  static final String[] QUERIES = {
    "(t w0)",
    "(t w7)",
    "(t nosuch)",
    "(p w0 w1)",
    "(p w1 w0)",
    "(p w0 w0)",
    "(p w0 w1 w2)",
    "(ps 2 w0 w1)",
    "(ps 3 w1 w0 w2)",
    "(ps 1 w0 w0)",
    "(ps 4 w2 w2 w3)",
    "(ps 6 w0 w1 w0)",
    "(p w3)",
    "(pre w1)",
    "(wc w?2)",
    "(re w[13]5?)",
    "(ts w4 w9 w30 zzz)",
    "(b 0 (+ (t w0)) (? (t w1)) (? (p w2 w3)))",
    "(b 0 (+ (t w1)) (- (t w2)))",
    "(b 2 (? (t w1)) (? (t w2)) (? (t w3)))",
    "(b 0 (+ (t w0)) (# (t w5)))",
    "(dismax 0.1 (t w1) (p w0 w1) (t w4))",
    "(const (t w6))",
    "(boost 2 (pre w2))",
    "(all)",
    "(b 0 (+ (t w0)) (# (r 0 200)))",
    "(r 100 150)",
    "(b 0 (+ (tf plain w1)) (? (t w1)))",
    "(tf tag w2)",
    "(tf plain w0)",
    "(pf plain 0 w0 w1)",
    "(pf plain 2 w1 w0)",
    "(b 0 (? (t w0)) (? (pre w0)))",
  };

  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]).resolve("matches_index");
    if (Files.exists(out)) {
      try (Stream<Path> s = Files.walk(out)) {
        s.sorted(Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
      }
    }
    Files.createDirectories(out);
    FieldType withOffsets = new FieldType(TextField.TYPE_NOT_STORED);
    withOffsets.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS);
    withOffsets.freeze();
    FieldType freqsOnly = new FieldType(TextField.TYPE_NOT_STORED);
    freqsOnly.setIndexOptions(IndexOptions.DOCS_AND_FREQS);
    freqsOnly.freeze();

    StringBuilder m = new StringBuilder();
    Random random = new Random(20260930L);
    try (Directory dir = FSDirectory.open(out)) {
      IndexWriterConfig cfg = new IndexWriterConfig(new StandardAnalyzer());
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);
      cfg.setMaxBufferedDocs(IndexWriterConfig.DISABLE_AUTO_FLUSH);
      cfg.setRAMBufferSizeMB(256);
      int id = 0;
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        for (int size : SEGMENTS) {
          for (int i = 0; i < size; i++, id++) {
            Document d = new Document();
            d.add(new StringField("id", Integer.toString(id), Field.Store.NO));
            d.add(new Field("body", body(random), withOffsets));
            if (id % 3 != 0) {
              d.add(new TextField("plain", body(random), Field.Store.NO));
            }
            if (id % 4 != 1) {
              d.add(new Field("tag", body(random), freqsOnly));
            }
            d.add(new LongPoint("r", id));
            w.addDocument(d);
          }
          w.commit();
        }
        for (int d = 0; d < SEGMENTS[0]; d += 9) {
          w.deleteDocuments(new Term("id", Integer.toString(d)));
        }
        w.commit();
      }

      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        IndexSearcher searcher = new IndexSearcher(reader);
        searcher.setQueryCache(null);
        m.append("query_count=").append(QUERIES.length).append('\n');
        for (int q = 0; q < QUERIES.length; q++) {
          m.append("q.").append(q).append('=').append(QUERIES[q]).append('\n');
          Query query = searcher.rewrite(parse(new Tokens(QUERIES[q])));
          Weight weight = searcher.createWeight(query, ScoreMode.COMPLETE_NO_SCORES, 1f);
          StringBuilder docs = new StringBuilder();
          for (LeafReaderContext ctx : reader.leaves()) {
            for (int doc = 0; doc < ctx.reader().maxDoc(); doc++) {
              Matches mt = weight.matches(ctx, doc);
              if (mt == null) {
                continue;
              }
              int global = ctx.docBase + doc;
              if (docs.length() > 0) docs.append(',');
              docs.append(global);
              m.append("m.").append(q).append('.').append(global).append('=').append(render(mt))
                  .append('\n');
            }
          }
          m.append("docs.").append(q).append('=').append(docs).append('\n');
        }

        // Named matches.
        String[][] named = {{"a", "(t w1)"}, {"b", "(p w0 w1)"}, {"c", "(pre w2)"}, {"d", "(r 20 40)"}};
        BooleanQuery.Builder nb = new BooleanQuery.Builder();
        for (String[] n : named) {
          m.append("named.q.").append(n[0]).append('=').append(n[1]).append('\n');
          nb.add(NamedMatches.wrapQuery(n[0], parse(new Tokens(n[1]))), BooleanClause.Occur.SHOULD);
        }
        Weight nw = searcher.createWeight(searcher.rewrite(nb.build()), ScoreMode.COMPLETE_NO_SCORES, 1f);
        for (LeafReaderContext ctx : reader.leaves()) {
          for (int doc = 0; doc < ctx.reader().maxDoc(); doc++) {
            Matches mt = nw.matches(ctx, doc);
            if (mt == null) {
              continue;
            }
            TreeSet<String> names = new TreeSet<>();
            for (NamedMatches n : NamedMatches.findNamedMatches(mt)) {
              names.add(n.getName());
            }
            m.append("named.").append(ctx.docBase + doc).append('=').append(String.join(",", names))
                .append('\n');
          }
        }
      }
    }
    Files.writeString(out.resolve("manifest.properties"), m.toString(), StandardCharsets.UTF_8);
  }

  static String body(Random r) {
    return GenMixedBooleanScoring.body(r);
  }

  static String render(Matches mt) throws IOException {
    if (mt == MatchesUtils.MATCH_WITH_NO_TERMS) {
      return "noterms";
    }
    StringBuilder b = new StringBuilder();
    for (String field : mt) {
      if (b.length() > 0) b.append(';');
      b.append(field).append('|');
      MatchesIterator it = mt.getMatches(field);
      boolean first = true;
      while (it != null && it.next()) {
        if (!first) b.append(',');
        first = false;
        b.append(it.startPosition())
            .append(':')
            .append(it.endPosition())
            .append(':')
            .append(it.startOffset())
            .append(':')
            .append(it.endOffset())
            .append(':')
            .append(it.getQuery().getClass().getSimpleName());
        if (it.getSubMatches() != null) {
          throw new AssertionError("a leaf iterator with sub-matches");
        }
      }
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

  static Query phrase(String field, int slop, Tokens t) {
    List<String> words = new ArrayList<>();
    while (!t.peek().equals(")")) {
      words.add(t.next());
    }
    return new PhraseQuery(slop, field, words.toArray(new String[0]));
  }

  static Query parse(Tokens t) {
    t.expect("(");
    String op = t.next();
    Query q;
    switch (op) {
      case "all" -> q = new MatchAllDocsQuery();
      case "t" -> q = new TermQuery(new Term("body", t.next()));
      case "tf" -> {
        String f = t.next();
        q = new TermQuery(new Term(f, t.next()));
      }
      case "r" -> q = LongPoint.newRangeQuery("r", Long.parseLong(t.next()), Long.parseLong(t.next()));
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
      case "p" -> q = phrase("body", 0, t);
      case "ps" -> {
        int slop = Integer.parseInt(t.next());
        q = phrase("body", slop, t);
      }
      case "pf" -> {
        String f = t.next();
        int slop = Integer.parseInt(t.next());
        q = phrase(f, slop, t);
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
