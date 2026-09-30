import org.apache.lucene.document.LongPoint;
import org.apache.lucene.index.Term;
import org.apache.lucene.queries.spans.SpanNearQuery;
import org.apache.lucene.queries.spans.SpanOrQuery;
import org.apache.lucene.queries.spans.SpanQuery;
import org.apache.lucene.queries.spans.SpanTermQuery;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.BoostQuery;
import org.apache.lucene.search.ConstantScoreQuery;
import org.apache.lucene.search.DisjunctionMaxQuery;
import org.apache.lucene.search.FieldExistsQuery;
import org.apache.lucene.search.FuzzyQuery;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.MatchNoDocsQuery;
import org.apache.lucene.search.MultiPhraseQuery;
import org.apache.lucene.search.PhraseQuery;
import org.apache.lucene.search.PrefixQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.RegexpQuery;
import org.apache.lucene.search.TermInSetQuery;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.WildcardQuery;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.automaton.ByteRunAutomaton;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;
import java.util.function.Supplier;

/**
 * {@code Query.visit} and {@code QueryVisitor} -- {@code termCollector}, {@code acceptField}, {@code
 * getSubVisitor}, {@code consumeTermsMatching} -- recorded from Lucene for
 * crates/lucene-search/tests/query_visitor_fixtures.rs. No index: each query is visited (not
 * rewritten) by {@code QueryVisitor.termCollector} and by a tracing visitor that logs every call
 * with its depth, the query's class, the terms, and a matcher's answers over a list of probe terms;
 * the trace is also taken with {@code acceptField} restricted to {@code body}. Traces are sorted:
 * {@code BooleanQuery} keeps each occurrence's clauses in hash order.
 */
public class GenQueryVisitor {
  static final String[] PROBES = {
    "w1", "w12", "w2", "w21", "w3", "x", "wa2", "w5", "w15", "w0", "w10", "a*b", "wz"
  };

  static final String[] QUERIES = {
    "(t w0)",
    "(tf plain w1)",
    "(p w0 w1 w2)",
    "(pf plain 1 w1 w0)",
    "(pre w1)",
    "(pre )",
    "(wc w?2)",
    "(wc w1)",
    "(wc a\\*b)",
    "(wc *)",
    "(re w[13]5?)",
    "(re w1)",
    "(re #)",
    "(ts w4 w9 w30)",
    "(ts w4)",
    "(ts w4 w4)",
    "(fz w1 1)",
    "(mp w0|w1 w2 w3|w4|w5)",
    "(st w1)",
    "(snear 1 true (st w0) (st w1))",
    "(sor (st w2) (snear 0 false (st w3) (st w4)))",
    "(all)",
    "(none)",
    "(r 1 5)",
    "(exists body)",
    "(b 0 (+ (t w0)) (# (t w1)) (? (p w2 w3)) (- (t w4)))",
    "(b 1 (? (tf plain w1)) (? (pre w2)) (- (b 0 (+ (t w9)))))",
    "(dismax 0.1 (t w1) (p w0 w1) (tf plain w4))",
    "(const (b 0 (+ (t w6)) (? (wc w?))))",
    "(boost 2 (pre w2))",
    "(b 0 (+ (boost 3 (t w1))) (# (const (tf plain w7))) (? (fz w2 2)))",
  };

  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]).resolve("query_visitor");
    Files.createDirectories(out);
    StringBuilder m = new StringBuilder();
    m.append("probes=").append(String.join(" ", PROBES)).append('\n');
    m.append("query_count=").append(QUERIES.length).append('\n');
    for (int i = 0; i < QUERIES.length; i++) {
      Query q = parse(new GenMatches.Tokens(QUERIES[i]));
      m.append("q.").append(i).append('=').append(QUERIES[i]).append('\n');
      Set<Term> terms = new TreeSet<>();
      q.visit(QueryVisitor.termCollector(terms));
      List<String> ts = new ArrayList<>();
      for (Term t : terms) ts.add(t.field() + ":" + t.text());
      m.append("terms.").append(i).append('=').append(String.join(",", ts)).append('\n');
      for (String only : new String[] {null, "body"}) {
        List<String> log = new ArrayList<>();
        q.visit(new Trace(0, log, only));
        Collections.sort(log);
        m.append(only == null ? "trace." : "trace_body.").append(i).append('=')
            .append(String.join(" | ", log)).append('\n');
      }
    }
    Files.writeString(out.resolve("cases.txt"), m.toString(), StandardCharsets.UTF_8);
  }

  static final class Trace extends QueryVisitor {
    final int depth;
    final List<String> log;
    final String only;

    Trace(int depth, List<String> log, String only) {
      this.depth = depth;
      this.log = log;
      this.only = only;
    }

    static String name(Query q) {
      return q.getClass().getSimpleName();
    }

    @Override
    public void consumeTerms(Query query, Term... terms) {
      List<String> ts = new ArrayList<>();
      for (Term t : terms) ts.add(t.field() + ":" + t.text());
      log.add("d" + depth + " terms " + name(query) + " " + String.join(",", ts));
    }

    @Override
    public void consumeTermsMatching(Query query, String field, Supplier<ByteRunAutomaton> automaton) {
      ByteRunAutomaton a = automaton.get();
      StringBuilder b = new StringBuilder();
      for (String p : PROBES) {
        byte[] bytes = p.getBytes(StandardCharsets.UTF_8);
        b.append(a.run(bytes, 0, bytes.length) ? '1' : '0');
      }
      log.add("d" + depth + " match " + name(query) + " " + field + " " + b);
    }

    @Override
    public void visitLeaf(Query query) {
      log.add("d" + depth + " leaf " + name(query));
    }

    @Override
    public boolean acceptField(String field) {
      return only == null || only.equals(field);
    }

    @Override
    public QueryVisitor getSubVisitor(BooleanClause.Occur occur, Query parent) {
      log.add("d" + depth + " sub " + occur + " " + name(parent));
      if (occur == BooleanClause.Occur.MUST_NOT) {
        return QueryVisitor.EMPTY_VISITOR;
      }
      return new Trace(depth + 1, log, only);
    }
  }

  static Query phrase(String field, int slop, GenMatches.Tokens t) {
    List<String> words = new ArrayList<>();
    while (!t.peek().equals(")")) {
      words.add(t.next());
    }
    return new PhraseQuery(slop, field, words.toArray(new String[0]));
  }

  static SpanQuery span(GenMatches.Tokens t) {
    t.expect("(");
    String op = t.next();
    SpanQuery q;
    switch (op) {
      case "st" -> q = new SpanTermQuery(new Term("body", t.next()));
      case "snear" -> {
        int slop = Integer.parseInt(t.next());
        boolean inOrder = Boolean.parseBoolean(t.next());
        List<SpanQuery> cs = new ArrayList<>();
        while (t.peek().equals("(")) cs.add(span(t));
        q = new SpanNearQuery(cs.toArray(new SpanQuery[0]), slop, inOrder);
      }
      case "sor" -> {
        List<SpanQuery> cs = new ArrayList<>();
        while (t.peek().equals("(")) cs.add(span(t));
        q = new SpanOrQuery(cs.toArray(new SpanQuery[0]));
      }
      default -> throw new IllegalArgumentException(op);
    }
    t.expect(")");
    return q;
  }

  static Query parse(GenMatches.Tokens t) {
    String op = t.toks.get(t.at + 1);
    if (op.equals("st") || op.equals("snear") || op.equals("sor")) {
      return span(t);
    }
    t.expect("(");
    t.next();
    Query q;
    switch (op) {
      case "all" -> q = new MatchAllDocsQuery();
      case "none" -> q = new MatchNoDocsQuery();
      case "exists" -> q = new FieldExistsQuery(t.next());
      case "t" -> q = new TermQuery(new Term("body", t.next()));
      case "tf" -> {
        String f = t.next();
        q = new TermQuery(new Term(f, t.next()));
      }
      case "r" -> q = LongPoint.newRangeQuery("r", Long.parseLong(t.next()), Long.parseLong(t.next()));
      case "pre" -> q = new PrefixQuery(new Term("body", t.peek().equals(")") ? "" : t.next()));
      case "wc" -> q = new WildcardQuery(new Term("body", t.next()));
      case "re" -> q = new RegexpQuery(new Term("body", t.next()));
      case "ts" -> {
        List<BytesRef> terms = new ArrayList<>();
        while (!t.peek().equals(")")) terms.add(new BytesRef(t.next()));
        q = new TermInSetQuery("body", terms);
      }
      case "fz" -> {
        String term = t.next();
        q = new FuzzyQuery(new Term("body", term), Integer.parseInt(t.next()));
      }
      case "mp" -> {
        MultiPhraseQuery.Builder b = new MultiPhraseQuery.Builder();
        while (!t.peek().equals(")")) {
          String[] alts = t.next().split("\\|");
          Term[] ts = new Term[alts.length];
          for (int k = 0; k < alts.length; k++) ts[k] = new Term("body", alts[k]);
          b.add(ts);
        }
        q = b.build();
      }
      case "p" -> q = phrase("body", 0, t);
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
        while (t.peek().equals("(")) ds.add(parse(t));
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
